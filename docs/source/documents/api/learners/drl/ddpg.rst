DDPG_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.ddpg_learner.DDPG_Learner(policy, optimizer, scheduler, device, model_dir, gamma, tau)

  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param tau: xxxxxx.
  :type tau: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.ddpg_learner.DDPG_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param rew_batch: xxxxxx.
  :type rew_batch: xxxxxx
  :param next_batch: xxxxxx.
  :type next_batch: xxxxxx
  :param terminal_batch: xxxxxx.
  :type terminal_batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.learners.policy_gradient.ddpg_learner.DDPG_Learner(policy, optimizer, scheduler, model_dir, gamma, tau)

  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param tau: xxxxxx.
  :type tau: xxxxxx

.. py:function::
  xuance.mindspore.learners.policy_gradient.ddpg_learner.DDPG_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param rew_batch: xxxxxx.
  :type rew_batch: xxxxxx
  :param next_batch: xxxxxx.
  :type next_batch: xxxxxx
  :param terminal_batch: xxxxxx.
  :type terminal_batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from xuance.torch.learners import *


        class DDPG_Learner(Learner):
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
                super(DDPG_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                rew_batch = torch.as_tensor(rew_batch, device=self.device)
                ter_batch = torch.as_tensor(terminal_batch, device=self.device)
                # critic update
                action_q = self.policy.Qaction(obs_batch, act_batch)
                # with torch.no_grad():
                target_q = self.policy.Qtarget(next_batch)
                backup = rew_batch + (1 - ter_batch) * self.gamma * target_q
                q_loss = F.mse_loss(action_q, backup.detach())
                self.optimizer[1].zero_grad()
                q_loss.backward()
                self.optimizer[1].step()

                # actor update
                policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = -policy_q.mean()
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


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *


        class DDPG_Learner(Learner):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(DDPG_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._mean = ms.ops.ReduceMean()

                def construct(self, x):
                    policy_q = self._backbone.Qpolicy(x)
                    loss_a = -self._mean(policy_q)
                    return loss_a

            class CriticNetWithLossCell(nn.Cell):
                def __init__(self, backbone, gamma):
                    super(DDPG_Learner.CriticNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._gamma = gamma
                    self._loss = nn.MSELoss()

                def construct(self, x, a, x_, q_target):
                    action_q = self._backbone.Qaction(x, a)
                    loss_q = self._loss(logits=action_q, labels=q_target)
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
                super(DDPG_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
                self.actor_train.set_train()
                self.critic_loss_net = self.CriticNetWithLossCell(policy, self.gamma)
                self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
                self.critic_train.set_train()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                rew_batch = Tensor(rew_batch)
                next_batch = Tensor(next_batch)
                ter_batch = Tensor(terminal_batch)

                target_q = self.policy.Qtarget(next_batch)
                backup = rew_batch + (1 - ter_batch) * self.gamma * target_q
                q_loss = self.critic_train(obs_batch, act_batch, next_batch, backup)
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
