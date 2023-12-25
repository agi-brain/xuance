SAC_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.policy_gradient.sac_learner.SAC_Learner(policy, optimizer, scheduler, device, model_dir, gamma, tau)

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
  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:function::
 xuance.torch.learners.policy_gradient.sac_learner.SAC_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param next_batch: A batch of next observations sampled from experience replay buffer.
  :type next_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.policy_gradient.sac_learner.SAC_Learner(policy, optimizer, device, model_dir, gamma, tau)

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
  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:function::
 xuance.tensorflow.learners.policy_gradient.sac_learner.SAC_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param next_batch: A batch of next observations sampled from experience replay buffer.
  :type next_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.policy_gradient.sac_learner.SAC_Learner(policy, optimizers, schedulers, model_dir, gamma, tau)

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
  :param tau: The soft update factor for the update of target networks.
  :type tau: float

.. py:function::
 xuance.mindspore.learners.policy_gradient.sac_learner.SAC_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param rew_batch: A batch of rewards sampled from experience replay buffer.
  :type rew_batch: np.ndarray
  :param next_batch: A batch of next observations sampled from experience replay buffer.
  :type next_batch: np.ndarray
  :param terminal_batch: A batch of terminal data sampled from experience replay buffer.
  :type terminal_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from xuance.torch.learners import *


        class SAC_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizers: Sequence[torch.optim.Optimizer],
                         schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01):
                self.tau = tau
                self.gamma = gamma
                super(SAC_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                rew_batch = torch.as_tensor(rew_batch, device=self.device)
                ter_batch = torch.as_tensor(terminal_batch, device=self.device)
                # critic update
                action_q = self.policy.Qaction(obs_batch, act_batch)
                # with torch.no_grad():
                log_pi_next, target_q = self.policy.Qtarget(next_batch)
                backup = rew_batch + (1-ter_batch) * self.gamma * (target_q - 0.01 * log_pi_next.reshape([-1]))
                q_loss = F.mse_loss(action_q, backup.detach())
                self.optimizer[1].zero_grad()
                q_loss.backward()
                self.optimizer[1].step()

                # actor update
                log_pi, policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = (0.01 * log_pi.reshape([-1]) - policy_q).mean()
                self.optimizer[0].zero_grad()
                p_loss.backward()
                self.optimizer[0].step()

                if self.scheduler is not None:
                    self.scheduler[0].step()
                    self.scheduler[1].step()

                self.policy.soft_update(self.tau)

                actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
                critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

                info = {
                    "Qloss": q_loss.item(),
                    "Ploss": p_loss.item(),
                    "Qvalue": action_q.mean().item(),
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr
                }

                return info







  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.learners import *


        class SAC_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizers: Sequence[tk.optimizers.Optimizer],
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01):
                self.tau = tau
                self.gamma = gamma
                super(SAC_Learner, self).__init__(policy, optimizers, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch)
                    rew_batch = tf.convert_to_tensor(rew_batch)
                    ter_batch = tf.convert_to_tensor(terminal_batch)

                    with tf.GradientTape() as tape:
                        # critic update
                        _, action_q = self.policy.Qaction(obs_batch, act_batch)
                        # with torch.no_grad():
                        _, log_pi_next, target_q = self.policy.Qtarget(next_batch)
                        target_q = tf.reshape(target_q, [-1])
                        log_pi_next = tf.reshape(log_pi_next, [-1])
                        backup = rew_batch + (1-ter_batch) * self.gamma * (target_q - 0.01 * log_pi_next)
                        y_true = tf.stop_gradient(tf.reshape(backup, [-1]))
                        y_pred = tf.reshape(action_q, [-1])
                        q_loss = tk.losses.mean_squared_error(y_true, y_pred)
                        gradients = tape.gradient(q_loss, self.policy.critic.trainable_variables)
                        self.optimizer[1].apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.critic.trainable_variables)
                            if grad is not None
                        ])

                    with tf.GradientTape() as tape:
                        # actor update
                        _, log_pi, policy_q = self.policy.Qpolicy(obs_batch)
                        p_loss = tf.reduce_mean(0.01 * log_pi - policy_q)
                        gradients = tape.gradient(p_loss, self.policy.actor.trainable_variables)
                        self.optimizer[0].apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.actor.trainable_variables)
                            if grad is not None
                        ])

                    self.policy.soft_update(self.tau)

                    actor_lr = self.optimizer[0]._decayed_lr(tf.float32)
                    critic_lr = self.optimizer[1]._decayed_lr(tf.float32)

                    info = {
                        "Qloss": q_loss.numpy(),
                        "Ploss": p_loss.numpy(),
                        "Qvalue": tf.reduce_mean(action_q).numpy(),
                        "actor_lr": actor_lr.numpy(),
                        "critic_lr": critic_lr.numpy()
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *
        from mindspore.nn.probability.distribution import Normal


        class SAC_Learner(Learner):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(SAC_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone

                def construct(self, x):
                    _, log_pi, policy_q = self._backbone.Qpolicy(x)
                    loss_a = (0.01 * log_pi.reshape([-1]) - policy_q).mean()
                    return loss_a

            class CriticNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(SAC_Learner.CriticNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()

                def construct(self, x, a, target):
                    _, action_q = self._backbone.Qaction(x, a)
                    loss_q = self._loss(logits=action_q, labels=target)
                    return loss_q

            def __init__(self,
                         policy: nn.Cell,
                         optimizers: nn.Optimizer,
                         schedulers: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01):
                self.tau = tau
                self.gamma = gamma
                super(SAC_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
                self.actor_train.set_train()
                self.critic_loss_net = self.CriticNetWithLossCell(policy)
                self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
                self.critic_train.set_train()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                rew_batch = Tensor(rew_batch)
                next_batch = Tensor(next_batch)
                ter_batch = Tensor(terminal_batch)

                _, log_pi_next, target_q = self.policy.Qtarget(next_batch)
                target = rew_batch + (1 - ter_batch) * self.gamma * (target_q - 0.01 * log_pi_next.reshape([-1]))

                q_loss = self.critic_train(obs_batch, act_batch, target)
                p_loss = self.actor_train(obs_batch)

                self.policy.soft_update(self.tau)

                actor_lr = self.scheduler['actor'](self.iterations).asnumpy()
                critic_lr = self.scheduler['critic'](self.iterations).asnumpy()

                info = {
                    "Qloss": q_loss.asnumpy(),
                    "Ploss": p_loss.asnumpy(),
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr
                }

                return info


