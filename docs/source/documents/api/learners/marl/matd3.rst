MATD3_Learner
=====================================

The implementation of the Multi-Agent Twin Delayed DDPG (MATD3) algorithm.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.matd3_learner.MATD3_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency, delay)

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
  :param delay: Delay factor for actor updates.
  :type delay: int

.. py:function::
  xuance.torch.learners.multi_agent_rl.matd3_learner.MATD3_Learner.update(sample)

  This method performs a single update step for the MATD3 learner based on the given experience sample.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.matd3_learner.MATD3_Learner(config, policy, optimizer, device, model_dir, gamma, sync_frequency, delay)

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
  :param delay: Delay factor for actor updates.
  :type delay: int

.. py:function::
  xuance.tensorflow.learners.multi_agent_rl.matd3_learner.MATD3_Learner.update(sample)

  This method performs a single update step for the MATD3 learner based on the given experience sample.

  :param sample: A dictionary containing necessary experience data that is sampled from experience replay buffer.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.matd3_learner.MATD3_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency, delay)

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
  :param delay: Delay factor for actor updates.
  :type delay: int

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.matd3_learner.MATD3_Learner.update(sample)

  This method performs a single update step for the MATD3 learner based on the given experience sample.

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
        Multi-Agent TD3
        """
        from xuance.torch.learners import *


        class MATD3_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: Sequence[torch.optim.Optimizer],
                         scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100,
                         delay: int = 3
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.delay = delay
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(MATD3_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic_A': optimizer[1],
                    'critic_B': optimizer[2]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic_A': scheduler[1],
                    'critic_B': scheduler[2]
                }

            def update(self, sample):
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                # train critic
                _, action_q = self.policy.Qaction(obs, actions, IDs)
                actions_next = self.policy.target_actor(obs_next, IDs)
                _, target_q = self.policy.Qtarget(obs_next, actions_next, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * target_q
                td_error = (action_q - q_target.detach()) * agent_mask
                loss_c = (td_error ** 2).sum() / agent_mask.sum()
                # loss_c = F.mse_loss(torch.tile(q_target.detach(), (1, 2)), action_q)
                self.optimizer['critic_B'].zero_grad()
                self.optimizer['critic_A'].zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
                self.optimizer['critic_A'].step()
                self.optimizer['critic_B'].step()
                if self.scheduler['critic_A'] is not None:
                    self.scheduler['critic_A'].step()
                    self.scheduler['critic_B'].step()

                # actor update
                if self.iterations % self.delay == 0:
                    _, actions_eval = self.policy(obs, IDs)
                    _, policy_q = self.policy.Qpolicy(obs, actions_eval, IDs)
                    p_loss = -policy_q.mean()
                    self.optimizer['actor'].zero_grad()
                    p_loss.backward()
                    self.optimizer['actor'].step()
                    if self.scheduler is not None:
                        self.scheduler['actor'].step()
                    self.policy.soft_update(self.tau)

                lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
                lr_c_A = self.optimizer['critic_A'].state_dict()['param_groups'][0]['lr']
                lr_c_B = self.optimizer['critic_B'].state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic_A": lr_c_A,
                    "learning_rate_critic_B": lr_c_B,
                    "loss_critic_A": loss_c.item(),
                    "loss_critic_B": loss_c.item()
                }
                if self.iterations % self.delay == 0:
                    info["loss_actor"] = p_loss.item()

                return info


  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        Multi-Agent TD3

        """
        from xuance.tensorflow.learners import *


        class MATD3_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: Sequence[tk.optimizers.Optimizer],
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100,
                         delay: int = 3
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.delay = delay
                self.sync_frequency = sync_frequency
                super(MATD3_Learner, self).__init__(config, policy, optimizer, device, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic': optimizer[1]
                }

            def update(self, sample):
                self.iterations += 1
                with tf.device(self.device):
                    obs = tf.convert_to_tensor(sample['obs'])
                    actions = tf.convert_to_tensor(sample['actions'])
                    obs_next = tf.convert_to_tensor(sample['obs_next'])
                    rewards = tf.convert_to_tensor(sample['rewards'])
                    terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
                    agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                            [-1, self.n_agents, 1])
                    IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))

                    # train critic
                    with tf.GradientTape() as tape:
                        _, action_q = self.policy.Qaction(obs, actions, IDs)
                        inputs_next = {"obs": obs_next, "ids": IDs}
                        actions_next = self.policy.target_actor(inputs_next)
                        _, target_q = self.policy.target_critic(obs_next, actions_next, IDs)
                        q_target = rewards + (1 - terminals) * self.args.gamma * target_q
                        y_pred = tf.reshape(action_q * agent_mask, [-1])
                        q_target = tf.tile(q_target, (1, 1, 2))
                        y_true = tf.reshape(tf.stop_gradient(q_target * agent_mask), [-1])
                        loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                        gradients = tape.gradient(loss_c, self.policy.critic_parameters)
                        self.optimizer['critic'].apply_gradients([
                            (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                            for (grad, var) in zip(gradients, self.policy.critic_parameters)
                            if grad is not None
                        ])

                    # actor update
                    with tf.GradientTape() as tape:
                        if self.iterations % self.delay == 0:
                            inputs = {"obs": obs, "ids": IDs}
                            _, actions_eval = self.policy(inputs)
                            _, policy_q = self.policy.critic(obs, actions_eval, IDs)
                            p_loss = -tf.reduce_mean(policy_q)
                            gradients = tape.gradient(p_loss, self.policy.actor_net.trainable_variables)
                            self.optimizer['actor'].apply_gradients([
                                (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                                for (grad, var) in zip(gradients, self.policy.actor_net.trainable_variables)
                                if grad is not None
                            ])
                            self.policy.soft_update(self.tau)

                    lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
                    lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

                    info = {
                        "learning_rate_actor": lr_a.numpy(),
                        "learning_rate_critic_A": lr_c.numpy(),
                        "loss_critic": loss_c.numpy(),
                        "predictQ": tf.math.reduce_mean(action_q).numpy()
                    }
                    if self.iterations % self.delay == 0:
                        info["loss_actor"] = p_loss.numpy()

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Multi-Agent TD3

        """
        from xuance.mindspore.learners import *


        class MATD3_Learner(LearnerMAS):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agents):
                    super(MATD3_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._mean = ms.ops.ReduceMean(keep_dims=True)
                    self.n_agents = n_agents

                def construct(self, bs, o, ids, agt_mask):
                    _, actions_eval = self._backbone(o, ids)
                    actions_n_eval = ms.ops.broadcast_to(actions_eval.view(bs, 1, -1), (-1, self.n_agents, -1))
                    _, policy_q = self._backbone.Qpolicy(o, actions_n_eval, ids)
                    loss_a = -policy_q.mean()
                    return loss_a

            class CriticNetWithLossCell_A(nn.Cell):
                def __init__(self, backbone):
                    super(MATD3_Learner.CriticNetWithLossCell_A, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()

                def construct(self, o, acts, ids, agt_mask, tar_q):
                    _, q_eval = self._backbone.Qaction_A(o, acts, ids)
                    td_error = (q_eval - tar_q) * agt_mask
                    loss_c = (td_error ** 2).sum() / agt_mask.sum()
                    return loss_c

            class CriticNetWithLossCell_B(nn.Cell):
                def __init__(self, backbone):
                    super(MATD3_Learner.CriticNetWithLossCell_B, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()

                def construct(self, o, acts, ids, agt_mask, tar_q):
                    _, q_eval = self._backbone.Qaction_B(o, acts, ids)
                    td_error = (q_eval - tar_q) * agt_mask
                    loss_c = (td_error ** 2).sum() / agt_mask.sum()
                    return loss_c

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: Sequence[nn.Optimizer],
                         scheduler: Sequence[nn .exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100,
                         delay: int = 3
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.delay = delay
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(MATD3_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic_A': optimizer[1],
                    'critic_B': optimizer[2]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic_A': scheduler[1],
                    'critic_B': scheduler[2]
                }
                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
                self.actor_train.set_train()
                self.critic_loss_net_A = self.CriticNetWithLossCell_A(policy)
                self.critic_train_A = nn.TrainOneStepCell(self.critic_loss_net_A, self.optimizer['critic_A'])
                self.critic_train_A.set_train()
                self.critic_loss_net_B = self.CriticNetWithLossCell_B(policy)
                self.critic_train_B = nn.TrainOneStepCell(self.critic_loss_net_B, self.optimizer['critic_B'])
                self.critic_train_B.set_train()

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                obs_next = Tensor(sample['obs_next'])
                rewards = Tensor(sample['rewards'])
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                # train critic
                actions_next = self.policy.target_actor(obs_next, IDs)
                actions_next_n = ms.ops.broadcast_to(actions_next.view(batch_size, 1, -1), (-1, self.n_agents, -1))
                _, target_q = self.policy.Qtarget(obs_next, actions_next_n, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * target_q

                actions_n = ms.ops.broadcast_to(actions.view(batch_size, 1, -1), (-1, self.n_agents, -1))
                loss_c_A = self.critic_train_A(obs, actions_n, IDs, agent_mask, q_target)
                loss_c_B = self.critic_train_B(obs, actions_n, IDs, agent_mask, q_target)

                # actor update
                if self.iterations % self.delay == 0:
                    p_loss = self.actor_train(batch_size, obs, IDs, agent_mask)
                    self.policy.soft_update(self.tau)

                lr_a = self.scheduler['actor'](self.iterations).asnumpy()
                lr_c_A = self.scheduler['critic_A'](self.iterations).asnumpy()
                lr_c_B = self.scheduler['critic_B'](self.iterations).asnumpy()

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic_A": lr_c_A,
                    "learning_rate_critic_B": lr_c_B,
                    "loss_critic_A": loss_c_A.asnumpy(),
                    "loss_critic_B": loss_c_B.asnumpy()
                }

                if self.iterations % self.delay == 0:
                    info["loss_actor"] = p_loss.asnumpy()

                return info


