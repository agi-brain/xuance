QTRAN_Learner
=====================================

An implementation of the QTRAN (Learning to Factorize with Transformation
for Cooperative Multi-Agent Reinforcement Learning) algorithm.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.qtran_learner.QTRAN_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  xuance.torch.learners.multi_agent_rl.qtran_learner.QTRAN_Learner.update(sample)

  Update the parameters of the QTRAN learner.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.qtran_learner.QTRAN_Learner(config, policy, optimizer, device, model_dir, gamma, sync_frequency)

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
  xuance.tensorflow.learners.multi_agent_rl.qtran_learner.QTRAN_Learner.update(sample)

  Update the parameters of the QTRAN learner.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.qtran_learner.QTRAN_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

  :param config: Provides hyper parameters.
  :type config: Namespace
  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param gamma: The discount factor.
  :type gamma: float
  :param sync_frequency: The frequency to synchronize the target networks.
  :type sync_frequency: int

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.qtran_learner.QTRAN_Learner.update(sample)

  Update the parameters of the QTRAN learner.

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
        QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
        Paper link:
        http://proceedings.mlr.press/v97/son19a/son19a.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *


        class QTRAN_Learner(LearnerMAS):
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
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(QTRAN_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

            def update(self, sample):
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                actions_onehot = self.onehot_action(actions, self.dim_act)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                hidden_n, _, q_eval = self.policy(obs, IDs)
                # get mask input
                actions_mask = agent_mask.repeat(1, 1, self.dim_act)
                hidden_mask = agent_mask.repeat(1, 1, hidden_n['state'].shape[-1])
                q_joint, v_joint = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                         actions_onehot * actions_mask)
                hidden_n_next, q_next_eval = self.policy.target_Q(obs_next.reshape([self.args.batch_size, self.n_agents, -1]), IDs)
                if self.args.double_q:
                    _, actions_next_greedy, _ = self.policy(obs_next, IDs)
                else:
                    actions_next_greedy = q_next_eval.argmax(dim=-1, keepdim=False)
                q_joint_next, _ = self.policy.target_qtran_net(hidden_n_next['state'] * hidden_mask,
                                                               self.onehot_action(actions_next_greedy,
                                                                                  self.dim_act) * actions_mask)
                y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next
                loss_td = self.mse_loss(q_joint, y_dqn.detach())

                action_greedy = q_eval.argmax(dim=-1, keepdim=False)  # \bar{u}
                q_eval_greedy_a = q_eval.gather(-1, action_greedy.long().reshape([self.args.batch_size, self.n_agents, 1]))
                q_tot_greedy = self.policy.q_tot(q_eval_greedy_a * agent_mask)
                q_joint_greedy_hat, _ = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                              self.onehot_action(action_greedy, self.dim_act) * actions_mask)
                error_opt = q_tot_greedy - q_joint_greedy_hat.detach() + v_joint
                loss_opt = torch.mean(error_opt ** 2)

                q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
                if self.args.agent == "QTRAN_base":
                    q_tot = self.policy.q_tot(q_eval_a * agent_mask)
                    q_joint_hat, _ = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                           actions_onehot * actions_mask)
                    error_nopt = q_tot - q_joint_hat.detach() + v_joint
                    error_nopt = error_nopt.clamp(max=0)
                    loss_nopt = torch.mean(error_nopt ** 2)
                elif self.args.agent == "QTRAN_alt":
                    q_tot_counterfactual = self.policy.qtran_net.counterfactual_values(q_eval, q_eval_a) * actions_mask
                    q_joint_hat_counterfactual = self.policy.qtran_net.counterfactual_values_hat(hidden_n['state'] * hidden_mask,
                                                                                                 actions_onehot * actions_mask)
                    error_nopt = q_tot_counterfactual - q_joint_hat_counterfactual.detach() + v_joint.unsqueeze(dim=-1).repeat(
                        1, self.n_agents, self.dim_act)
                    error_nopt_min = torch.min(error_nopt, dim=-1).values
                    loss_nopt = torch.mean(error_nopt_min ** 2)
                else:
                    raise ValueError("Mixer {} not recognised.".format(self.args.agent))

                # calculate the loss function
                loss = loss_td + self.args.lambda_opt * loss_opt + self.args.lambda_nopt * loss_nopt
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_td": loss_td.item(),
                    "loss_opt": loss_opt.item(),
                    "loss_nopt": loss_nopt.item(),
                    "loss": loss.item(),
                    "predictQ": q_eval_a.mean().item()
                }

                return info

  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
        Paper link:
        http://proceedings.mlr.press/v97/son19a/son19a.pdf
        Implementation: TensorFlow 2.X
        """
        from xuance.tensorflow.learners import *


        class QTRAN_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(QTRAN_Learner, self).__init__(config, policy, optimizer, device, model_dir)

            def update(self, sample):
                self.iterations += 1
                with tf.device(self.device):
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
                    actions_onehot = self.onehot_action(actions, self.dim_act)
                    obs_next = tf.convert_to_tensor(sample['obs_next'])
                    rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
                    terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'].all(axis=-1, keepdims=True), dtype=tf.float32), [-1, 1])
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                            [-1, self.n_agents, 1])
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
                    batch_size = obs.shape[0]

                    with tf.GradientTape() as tape:
                        inputs_policy = {"obs": obs, "ids": IDs}
                        hidden_n, _, q_eval = self.policy(inputs_policy)
                        # get mask input
                        actions_mask = tf.tile(agent_mask, multiples=(1, 1, self.dim_act))
                        hidden_mask = tf.tile(agent_mask, multiples=(1, 1, hidden_n.shape[-1]))
                        q_joint, v_joint = self.policy.qtran_net(hidden_n * hidden_mask,
                                                                 actions_onehot * actions_mask)
                        inputs_target = {"obs": obs_next, "ids": IDs}
                        hidden_n_next, q_next_eval = self.policy.target_Q(inputs_target)
                        if self.args.double_q:
                            inputs_target = {"obs": obs_next, "ids": IDs}
                            _, actions_next_greedy, _ = self.policy(inputs_target)
                        else:
                            actions_next_greedy = tf.argmax(q_next_eval, axis=-1)
                        q_joint_next, _ = self.policy.target_qtran_net(hidden_n_next * hidden_mask,
                                                                       self.onehot_action(actions_next_greedy,
                                                                                          self.dim_act) * actions_mask)
                        y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next
                        y_dqn = tf.stop_gradient(tf.reshape(y_dqn, [-1]))
                        q_joint = tf.reshape(q_joint, [-1])
                        loss_td = tk.losses.mean_squared_error(y_dqn, q_joint)

                        action_greedy = tf.argmax(q_eval, axis=-1)  # \bar{u}
                        q_eval_greedy_a = tf.gather(q_eval, tf.reshape(action_greedy, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                        q_tot_greedy = self.policy.q_tot(q_eval_greedy_a * agent_mask)
                        q_joint_greedy_hat, _ = self.policy.qtran_net(hidden_n * hidden_mask,
                                                                      self.onehot_action(action_greedy, self.dim_act) * actions_mask)
                        error_opt = q_tot_greedy - tf.stop_gradient(q_joint_greedy_hat) + v_joint
                        loss_opt = tf.reduce_mean(error_opt ** 2)

                        q_eval_a = tf.gather(q_eval, tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                        if self.args.agent == "QTRAN_base":
                            q_tot = self.policy.q_tot(q_eval_a * agent_mask)
                            q_joint_hat, _ = self.policy.qtran_net(hidden_n * hidden_mask,
                                                                   actions_onehot * actions_mask)
                            error_nopt = q_tot - tf.stop_gradient(q_joint_hat) + v_joint
                            error_nopt = tf.clip_by_value(error_nopt, clip_value_min=-1e10, clip_value_max=0)
                            loss_nopt = tf.reduce_mean(error_nopt ** 2)
                        elif self.args.agent == "QTRAN_alt":
                            q_tot_counterfactual = self.policy.qtran_net.counterfactual_values(q_eval, q_eval_a) * actions_mask
                            q_joint_hat_counterfactual = self.policy.qtran_net.counterfactual_values_hat(hidden_n * hidden_mask,
                                                                                                         actions_onehot * actions_mask)
                            v_joint_repeat = tf.tile(tf.expand_dims(v_joint, axis=-1), multiples=(1, self.n_agents, self.dim_act))
                            error_nopt = q_tot_counterfactual - tf.stop_gradient(q_joint_hat_counterfactual) + v_joint_repeat
                            error_nopt_min = tf.reduce_min(error_nopt, axis=-1)
                            loss_nopt = tf.reduce_mean(error_nopt_min ** 2)
                        else:
                            raise ValueError("Mixer {} not recognised.".format(self.args.agent))

                        # calculate the loss function
                        loss = loss_td + self.args.lambda_opt * loss_opt + self.args.lambda_nopt * loss_nopt
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
                            "loss_td": loss_td.numpy(),
                            "loss_opt": loss_opt.numpy(),
                            "loss_nopt": loss_nopt.numpy(),
                            "loss": loss.numpy(),
                            "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
                        }

                        return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
        Paper link:
        http://proceedings.mlr.press/v97/son19a/son19a.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class QTRAN_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, dim_act, n_agents, agent_name, lambda_opt, lambda_nopt):
                    super(QTRAN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self.dim_act = dim_act
                    self.n_agents = n_agents
                    self.agent = agent_name
                    self._lambda_opt = lambda_opt
                    self._lambda_nopt = lambda_nopt

                    self._expand_dims = ops.ExpandDims()
                    self._onehot = ms.ops.OneHot()

                def construct(self, o, ids, a, a_onehot, agt_mask, act_mask, hidden_mask, y_dqn):
                    _, hidden_state, _, q_eval = self._backbone(o, ids)
                    q_joint, v_joint = self._backbone.qtran_net(hidden_state * hidden_mask,
                                                                a_onehot * act_mask)
                    loss_td = ((q_joint - y_dqn) ** 2).sum() / agt_mask.sum()

                    action_greedy = q_eval.argmax(axis=-1).astype(ms.int32)  # \bar{u}
                    q_eval_greedy_a = GatherD()(q_eval, -1, action_greedy.view(-1, self.n_agents, 1))
                    q_tot_greedy = self._backbone.q_tot(q_eval_greedy_a * agt_mask)
                    q_joint_greedy_hat, _ = self._backbone.qtran_net(hidden_state * hidden_mask,
                                                                     self._onehot(action_greedy, self.dim_act,
                                                                                  ms.Tensor(1.0, ms.float32),
                                                                                  ms.Tensor(0.0, ms.float32)) * act_mask)
                    error_opt = q_tot_greedy - q_joint_greedy_hat + v_joint
                    loss_opt = (error_opt ** 2).mean()

                    q_eval_a = GatherD()(q_eval, -1, a)
                    if self.agent == "QTRAN_base":
                        q_tot = self._backbone.q_tot(q_eval_a * agt_mask)
                        q_joint_hat, _ = self._backbone.qtran_net(hidden_state * hidden_mask, a_onehot * act_mask)
                        error_nopt = q_tot - q_joint_hat + v_joint
                        error_nopt = ops.clip_by_value(error_nopt, clip_value_max=ms.Tensor(0.0, ms.float32))
                        loss_nopt = (error_nopt ** 2).mean()
                    elif self.agent == "QTRAN_alt":
                        q_tot_counterfactual = self._backbone.qtran_net.counterfactual_values(q_eval, q_eval_a) * act_mask
                        q_joint_hat_counterfactual = self._backbone.qtran_net.counterfactual_values_hat(
                            hidden_state * hidden_mask, a_onehot * act_mask)
                        error_nopt = q_tot_counterfactual - q_joint_hat_counterfactual + ops.broadcast_to(
                            self._expand_dims(v_joint, -1), (-1, -1, self.dim_act))
                        error_nopt_min = error_nopt.min(axis=-1)
                        loss_nopt = (error_nopt_min ** 2).mean()
                    else:
                        raise ValueError("Mixer {} not recognised.".format(self.args.agent))

                    loss = loss_td + self._lambda_opt * loss_opt + self._lambda_nopt * loss_nopt
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
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(QTRAN_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self._mean = ops.ReduceMean(keep_dims=False)
                self.loss_net = self.PolicyNetWithLossCell(policy, self.dim_act, self.n_agents, self.args.agent,
                                                           self.args.lambda_opt, self.args.lambda_nopt)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                actions_onehot = self.onehot_action(actions, self.dim_act)
                actions = actions.view(-1, self.n_agents, 1).astype(ms.int32)
                obs_next = Tensor(sample['obs_next'])
                rewards = self._mean(Tensor(sample['rewards']), 1)
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=True).astype(ms.float32)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                actions_mask = ops.broadcast_to(agent_mask, (-1, -1, int(self.dim_act)))
                hidden_mask = ops.broadcast_to(agent_mask, (-1, -1, self.policy.representation_info_shape['state'][0]))

                _, hidden_state_next, q_next_eval = self.policy.target_Q(obs_next.view(batch_size, self.n_agents, -1), IDs)
                if self.args.double_q:
                    _, _, actions_next_greedy, _ = self.policy(obs_next, IDs)
                else:
                    actions_next_greedy = q_next_eval.argmax(axis=-1, keepdim=False)
                q_joint_next, _ = self.policy.target_qtran_net(hidden_state_next * hidden_mask,
                                                               self.onehot_action(actions_next_greedy,
                                                                                  self.dim_act) * actions_mask)
                y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next

                # calculate the loss function
                loss = self.policy_train(obs, IDs, actions, actions_onehot, agent_mask, actions_mask, hidden_mask, y_dqn)
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss": loss.asnumpy()
                }

                return info


