WQMIX_Learner
=====================================

An implementation of the Weighted QMIX (WQMIX) algorithm.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.torch.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner.update(sample)

  Update the parameters of the model.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner.update_recurrent(sample)

  Update the parameters of the model with recurrent neural networks.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner(config, policy, optimizer, device, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner.update(sample)

  Update the parameters of the model.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.wqmix_learner.WQMIX_Learner.update(sample)

  Update the parameters of the model.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        """
        Weighted QMIX
        Paper link:
        https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *


        class WQMIX_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.alpha = config.alpha
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(WQMIX_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

            def update(self, sample):
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                state_next = torch.Tensor(sample['state_next']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
                terminals = torch.Tensor(sample['terminals']).all(dim=1, keepdims=True).float().to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                batch_size = actions.shape[0]
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                # calculate Q_tot
                _, action_max, q_eval = self.policy(obs, IDs)
                action_max = action_max.unsqueeze(-1)
                q_eval_a = q_eval.gather(-1, actions.long().reshape(batch_size, self.n_agents, 1))
                q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)

                # calculate centralized Q
                q_eval_centralized = self.policy.q_centralized(obs, IDs).gather(-1, action_max.long())
                q_tot_centralized = self.policy.q_feedforward(q_eval_centralized * agent_mask, state)

                # calculate y_i
                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(obs_next, IDs)
                    action_next_greedy = action_next_greedy.unsqueeze(-1)
                else:
                    q_next_eval = self.policy.target_Q(obs_next, IDs)
                    action_next_greedy = q_next_eval.argmax(dim=-1, keepdim=True)
                q_eval_next_centralized = self.policy.target_q_centralized(obs_next, IDs).gather(-1, action_next_greedy)
                q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized * agent_mask, state_next)

                target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
                td_error = q_tot_eval - target_value.detach()

                # calculate weights
                ones = torch.ones_like(td_error)
                w = ones * self.alpha
                if self.args.agent == "CWQMIX":
                    condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
                    condition_2 = target_value > q_tot_centralized
                    conditions = condition_1 | condition_2
                    w = torch.where(conditions, ones, w)
                elif self.args.agent == "OWQMIX":
                    condition = td_error < 0
                    w = torch.where(condition, ones, w)
                else:
                    AttributeError("You have assigned an unexpected WQMIX learner!")

                # calculate losses and train
                loss_central = self.mse_loss(q_tot_centralized, target_value.detach())
                loss_qmix = (w.detach() * (td_error ** 2)).mean()
                loss = loss_qmix + loss_central
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Qmix": loss_qmix.item(),
                    "loss_central": loss_central.item(),
                    "loss": loss.item(),
                    "predictQ": q_tot_eval.mean().item()
                }

                return info

            def update_recurrent(self, sample):
                """
                Update the parameters of the model with recurrent neural networks.
                """
                self.iterations += 1
                state = torch.Tensor(sample['state']).to(self.device)
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1, keepdims=False).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().to(self.device)
                avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
                filled = torch.Tensor(sample['filled']).float().to(self.device)
                batch_size = actions.shape[0]
                episode_length = actions.shape[2]
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
                    self.device)

                # calculate Q_tot
                rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
                _, actions_greedy, q_eval = self.policy(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                        IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                        *rnn_hidden,
                                                        avail_actions=avail_actions.reshape(-1, episode_length + 1, self.dim_act))
                q_eval = q_eval[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
                actions_greedy = actions_greedy.reshape(batch_size, self.n_agents, episode_length + 1, 1).detach()
                q_eval_a = q_eval.gather(-1, actions.long().reshape(batch_size, self.n_agents, episode_length, 1))
                q_eval_a = q_eval_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1])

                # calculate centralized Q
                q_eval_centralized = self.policy.q_centralized(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                               IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                               *rnn_hidden)
                q_eval_centralized = q_eval_centralized[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
                q_eval_centralized_a = q_eval_centralized.gather(-1, actions_greedy[:, :, :-1].long())
                q_eval_centralized_a = q_eval_centralized_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state[:, :-1])

                # calculate y_i
                target_rnn_hidden = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
                if self.args.double_q:
                    action_next_greedy = actions_greedy[:, :, 1:]
                else:
                    _, q_next = self.policy.target_Q(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                     IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                     *target_rnn_hidden)
                    q_next = q_next[:, 1:].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
                    q_next[avail_actions[:, :, 1:] == 0] = -9999999
                    action_next_greedy = q_next.argmax(dim=-1, keepdim=True)
                q_eval_next_centralized = self.policy.target_q_centralized(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                                           IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                                           *target_rnn_hidden)
                q_eval_next_centralized = q_eval_next_centralized[:, 1:].reshape(batch_size, self.n_agents, episode_length,
                                                                              self.dim_act)
                q_eval_next_centralized_a = q_eval_next_centralized.gather(-1, action_next_greedy)
                q_eval_next_centralized_a = q_eval_next_centralized_a.transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state[:, 1:])

                rewards = rewards.reshape(-1, 1)
                terminals = terminals.reshape(-1, 1)
                filled = filled.reshape(-1, 1)
                target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
                td_error = q_tot_eval - target_value.detach()
                td_error *= filled

                # calculate weights
                ones = torch.ones_like(td_error)
                w = ones * self.alpha
                if self.args.agent == "CWQMIX":
                    actions_greedy = actions_greedy[:, :, :-1]
                    condition_1 = (actions_greedy == actions.reshape([-1, self.n_agents, episode_length, 1])).all(dim=1)
                    condition_1 = condition_1.reshape(-1, 1)
                    condition_2 = target_value > q_tot_centralized
                    conditions = condition_1 | condition_2
                    w = torch.where(conditions, ones, w)
                elif self.args.agent == "OWQMIX":
                    condition = td_error < 0
                    w = torch.where(condition, ones, w)
                else:
                    AttributeError("You have assigned an unexpected WQMIX learner!")

                # calculate losses and train
                error_central = (q_tot_centralized - target_value.detach()) * filled
                loss_central = (error_central ** 2).sum() / filled.sum()
                loss_qmix = (w.detach() * (td_error ** 2)).sum() / filled.sum()
                loss = loss_qmix + loss_central
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Qmix": loss_qmix.item(),
                    "loss_central": loss_central.item(),
                    "loss": loss.item(),
                    "predictQ": q_tot_eval.mean().item()
                }

                return info

  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        Weighted QMIX
        Paper link:
        https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
        Implementation: TensorFlow 2.X
        """
        from xuance.tensorflow.learners import *


        class WQMIX_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.alpha = config.alpha
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(WQMIX_Learner, self).__init__(config, policy, optimizer, device, model_dir)

            def update(self, sample):
                self.iterations += 1
                with tf.device(self.device):
                    state = tf.convert_to_tensor(sample['state'])
                    state_next = tf.convert_to_tensor(sample['state_next'])
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
                    obs_next = tf.convert_to_tensor(sample['obs_next'])
                    rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
                    terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'].all(axis=-1, keepdims=True), dtype=tf.float32), [-1, 1])
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                            [-1, self.n_agents, 1])
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
                    batch_size = obs.shape[0]

                    with tf.GradientTape() as tape:
                        # calculate Q_tot
                        inputs_policy = {"obs": obs, "ids": IDs}
                        _, action_max, q_eval = self.policy(inputs_policy)
                        action_max = tf.expand_dims(action_max, axis=-1)
                        q_eval_a = tf.gather(q_eval, indices=tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                        q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)

                        # calculate centralized Q
                        q_eval_centralized = tf.gather(self.policy.q_centralized(inputs_policy), action_max, axis=-1, batch_dims=-1)
                        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized*agent_mask, state)

                        # calculate y_i
                        inputs_target = {"obs": obs_next, "ids": IDs}
                        if self.args.double_q:
                            _, action_next_greedy, _ = self.policy(inputs_target)
                            action_next_greedy = tf.expand_dims(action_next_greedy, axis=-1)
                        else:
                            q_next_eval = self.policy.target_Q(inputs_target)
                            action_next_greedy = tf.argmax(q_next_eval, axis=-1)
                        q_eval_next_centralized = tf.gather(self.policy.target_q_centralized(inputs_target), action_next_greedy, axis=-1, batch_dims=-1)
                        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized*agent_mask, state_next)

                        target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
                        td_error = q_tot_eval - tf.stop_gradient(target_value)

                        # calculate weights
                        ones = tf.ones_like(td_error)
                        w = ones * self.alpha
                        if self.args.agent == "CWQMIX":
                            condition_1 = tf.cast((action_max == tf.reshape(actions, [-1, self.n_agents, 1])), dtype=tf.float32)
                            condition_1 = tf.reduce_all(tf.cast(condition_1 * agent_mask, dtype=tf.bool), axis=1)
                            condition_2 = target_value > q_tot_centralized
                            conditions = condition_1 | condition_2
                            w = tf.where(conditions, ones, w)
                        elif self.args.agent == "OWQMIX":
                            condition = td_error < 0
                            w = tf.where(condition, ones, w)
                        else:
                            AttributeError("You have assigned an unexpected WQMIX learner!")

                        # calculate losses and train
                        y_true = tf.stop_gradient(tf.reshape(target_value, [-1]))
                        y_pred = tf.reshape(q_tot_centralized, [-1])
                        loss_central = tk.losses.mean_squared_error(y_true, y_pred)
                        loss_qmix = tf.reduce_mean((w * (td_error ** 2)))
                        loss = loss_qmix + loss_central
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])

                    if self.iterations % self.sync_frequency == 0:
                        self.policy.copy_target()

                    lr = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "learning_rate": lr.numpy(),
                        "loss_Qmix": loss_qmix.numpy(),
                        "loss_central": loss_central.numpy(),
                        "loss": loss.numpy(),
                        "predictQ": tf.math.reduce_mean(q_tot_eval).numpy()
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Weighted QMIX
        Paper link:
        https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class WQMIX_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agent, agent_name, alpha):
                    super(WQMIX_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self.n_agent = n_agent
                    self.agent = agent_name
                    self._backbone = backbone
                    self.alpha = alpha

                def construct(self, s, o, ids, a, label, agt_mask):
                    # calculate Q_tot
                    _, action_max, q_eval = self._backbone(o, ids)
                    action_max = action_max.view(-1, self.n_agent, 1)
                    q_eval_a = GatherD()(q_eval, -1, a)
                    q_tot_eval = self._backbone.Q_tot(q_eval_a * agt_mask, s)

                    # calculate centralized Q
                    q_centralized_eval = self._backbone.q_centralized(o, ids)
                    q_centralized_eval_a = GatherD()(q_centralized_eval, -1, action_max)
                    q_tot_centralized = self._backbone.q_feedforward(q_centralized_eval_a * agt_mask, s)
                    td_error = q_tot_eval - label

                    # calculate weights
                    ones = ops.ones_like(td_error)
                    w = ones * self.alpha
                    if self.agent == "CWQMIX":
                        condition_1 = ((action_max == a).astype(ms.float32) * agt_mask).astype(ms.bool_).all(axis=1)
                        condition_2 = label > q_tot_centralized
                        conditions = ops.logical_or(condition_1, condition_2)
                        w = ms.numpy.where(conditions, ones, w)
                    elif self.agent == "OWQMIX":
                        condition = td_error < 0
                        w = ms.numpy.where(condition, ones, w)
                    else:
                        AttributeError("You have assigned an unexpected WQMIX learner!")

                    loss_central = ((q_tot_centralized - label) ** 2).sum() / agt_mask.sum()
                    loss_qmix = (w * (td_error ** 2)).mean()
                    loss = loss_qmix + loss_central
                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.alpha = config.alpha
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(WQMIX_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                # build train net
                self._mean = ops.ReduceMean(keep_dims=False)
                self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, self.args.agent, self.alpha)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, sample):
                self.iterations += 1
                state = Tensor(sample['state'])
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions']).view(-1, self.n_agents, 1).astype(ms.int32)
                state_next = Tensor(sample['state_next'])
                obs_next = Tensor(sample['obs_next'])
                rewards = self._mean(Tensor(sample['rewards']), 1)
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=True)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))
                # calculate y_i
                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(obs_next, IDs)
                    action_next_greedy = self.expand_dims(action_next_greedy, -1).astype(ms.int32)
                else:
                    q_next_eval = self.policy.target_Q(obs_next, IDs)
                    action_next_greedy = q_next_eval.argmax(axis=-1, keepdims=True)
                q_eval_next_centralized = GatherD()(self.policy.target_q_centralized(obs_next, IDs), -1, action_next_greedy)
                q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized*agent_mask, state_next)

                target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized

                # calculate losses and train
                loss = self.policy_train(state, obs, IDs, actions, target_value, agent_mask)
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss": loss.asnumpy()
                }

                return info


