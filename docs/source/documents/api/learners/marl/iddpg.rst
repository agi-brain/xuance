IDDPG_Learner
=====================================

An Independent Deep Deterministic Policy Gradient (IDDPG) learner.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  xuance.torch.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner.update(sample)

  Update the IDDPG learner with a batch of samples.

  :param sample: A dictionary containing current observations, actions taken in the current state,
                    observations of the next state, rewards, binary indicators of episode terminations,
                    binary mask indicating which agents are active.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner(config, policy, optimizer, device, model_dir, gamma, sync_frequency)

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
  xuance.tensorflow.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner.update(sample)

  Update the IDDPG learner with a batch of samples.

  :param sample: A dictionary containing current observations, actions taken in the current state,
                    observations of the next state, rewards, binary indicators of episode terminations,
                    binary mask indicating which agents are active.
  :type sample: dict
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

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
  xuance.mindspore.learners.multi_agent_rl.iddpg_learner.IDDPG_Learner.update(sample)

  Update the IDDPG learner with a batch of samples.

  :param sample: A dictionary containing current observations, actions taken in the current state,
                    observations of the next state, rewards, binary indicators of episode terminations,
                    binary mask indicating which agents are active.
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
        Independent Deep Deterministic Policy Gradient (IDDPG)
        Implementation: Pytorch
        """
        from xuance.torch.learners import *


        class IDDPG_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: Sequence[torch.optim.Optimizer],
                         scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(IDDPG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic': optimizer[1]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic': scheduler[1]
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

                q_eval = self.policy.critic(obs, actions, IDs)
                q_next = self.policy.target_critic(obs_next, self.policy.target_actor(obs_next, IDs), IDs)
                q_target = rewards + (1-terminals) * self.args.gamma * q_next

                # calculate the loss function
                _, actions_eval = self.policy(obs, IDs)
                loss_a = -(self.policy.critic(obs, actions_eval, IDs) * agent_mask).sum() / agent_mask.sum()
                self.optimizer['actor'].zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.grad_clip_norm)
                self.optimizer['actor'].step()
                if self.scheduler['actor'] is not None:
                    self.scheduler['actor'].step()

                td_error = (q_eval - q_target.detach()) * agent_mask
                loss_c = (td_error ** 2).sum() / agent_mask.sum()
                self.optimizer['critic'].zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
                self.optimizer['critic'].step()
                if self.scheduler['critic'] is not None:
                    self.scheduler['critic'].step()

                self.policy.soft_update(self.tau)

                lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
                lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic": lr_c,
                    "loss_actor": loss_a.item(),
                    "loss_critic": loss_c.item(),
                    "predictQ": q_eval.mean().item()
                }

                return info






  .. group-tab:: TensorFlow

    .. code-block:: python

        """
        Independent Deep Deterministic Policy Gradient (IDDPG)
        Implementation: TensorFlow 2.X
        """
        from xuance.tensorflow.learners import *


        class IDDPG_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: tk.Model,
                         optimizer: Sequence[tk.optimizers.Optimizer],
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.sync_frequency = sync_frequency
                super(IDDPG_Learner, self).__init__(config, policy, optimizer, device, model_dir)
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
                    batch_size = obs.shape[0]

                    # train actor
                    with tf.GradientTape() as tape:
                        # calculate the loss function
                        inputs = {"obs": obs, "ids": IDs}
                        _, actions_eval = self.policy(inputs)
                        loss_a = -tf.reduce_sum(self.policy.critic(obs, actions_eval, IDs) * agent_mask) / tf.reduce_sum(agent_mask)
                        gradients = tape.gradient(loss_a, self.policy.parameters_actor)
                        self.optimizer['actor'].apply_gradients([
                            (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                            for (grad, var) in zip(gradients, self.policy.parameters_actor)
                            if grad is not None
                        ])

                    # train critic
                    with tf.GradientTape() as tape:
                        inputs_next = {"obs": obs_next, "ids": IDs}
                        q_eval = self.policy.critic(obs, actions, IDs)
                        q_next = self.policy.target_critic(obs_next, self.policy.target_actor(inputs_next), IDs)
                        q_target = rewards + (1 - terminals) * self.args.gamma * q_next
                        y_pred = tf.reshape(q_eval * agent_mask, [-1])
                        y_true = tf.stop_gradient(tf.reshape(q_target * agent_mask, [-1]))
                        loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                        gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                        self.optimizer['critic'].apply_gradients([
                            (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                            for (grad, var) in zip(gradients, self.policy.parameters_critic)
                            if grad is not None
                        ])

                    self.policy.soft_update(self.tau)

                    lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
                    lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

                    info = {
                        "learning_rate_actor": lr_a.numpy(),
                        "learning_rate_critic": lr_c.numpy(),
                        "loss_actor": loss_a.numpy(),
                        "loss_critic": loss_c.numpy(),
                        "predictQ": tf.math.reduce_mean(q_eval).numpy()
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Independent Deep Deterministic Policy Gradient (IDDPG)
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class IDDPG_Learner(LearnerMAS):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(IDDPG_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._mean = ms.ops.ReduceMean(keep_dims=True)

                def construct(self, o, ids, agt_mask):
                    _, actions_eval = self._backbone(o, ids)
                    loss_a = -(self._backbone.critic(o, actions_eval, ids) * agt_mask).sum() / agt_mask.sum()
                    return loss_a

            class CriticNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(IDDPG_Learner.CriticNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()

                def construct(self, o, a, ids, agt_mask, tar_q):
                    q_eval = self._backbone.critic(o, a, ids)
                    td_error = (q_eval - tar_q) * agt_mask
                    loss_c = (td_error ** 2).sum() / agt_mask.sum()
                    return loss_c

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: Sequence[nn.Optimizer],
                         scheduler: Sequence[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.tau = config.tau
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(IDDPG_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.optimizer = {
                    'actor': optimizer[0],
                    'critic': optimizer[1]
                }
                self.scheduler = {
                    'actor': scheduler[0],
                    'critic': scheduler[1]
                }

                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
                self.actor_train.set_train()
                self.critic_loss_net = self.CriticNetWithLossCell(policy)
                self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, self.optimizer['critic'])
                self.critic_train.set_train()

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
                # calculate target values
                q_next = self.policy.target_critic(obs_next, self.policy.target_actor(obs_next, IDs), IDs)
                q_target = rewards + (1-terminals) * self.args.gamma * q_next

                # calculate the loss and train
                loss_a = self.actor_train(obs, IDs, agent_mask)
                loss_c = self.critic_train(obs, actions, IDs, agent_mask, q_target)
                self.policy.soft_update(self.tau)

                lr_a = self.scheduler['actor'](self.iterations).asnumpy()
                lr_c = self.scheduler['critic'](self.iterations).asnumpy()

                info = {
                    "learning_rate_actor": lr_a,
                    "learning_rate_critic": lr_c,
                    "loss_actor": loss_a.asnumpy(),
                    "loss_critic": loss_c.asnumpy()
                }

                return info

