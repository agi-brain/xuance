TD3_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.td3_learner.TD3_Learner(policy, optimizer, scheduler, device, model_dir, gamma, tau, delay)

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
  :param delay: xxxxxx.
  :type delay: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.td3_learner.TD3_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

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
  xuance.mindspore.learners.policy_gradient.td3_learner.TD3_Learner(policy, optimizer, scheduler, model_dir, gamma, tau, delay)

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
  :param delay: xxxxxx.
  :type delay: xxxxxx

.. py:function::
  xuance.mindspore.learners.policy_gradient.td3_learner.TD3_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

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

        # TD3 add three tricks to DDPG:
        # 1. noisy action in target actor
        # 2. double critic network
        # 3. delayed actor update
            from xuance.torch.learners import *


            class TD3_Learner(Learner):
                def __init__(self,
                             policy: nn.Module,
                             optimizers: Sequence[torch.optim.Optimizer],
                             schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                             device: Optional[Union[int, str, torch.device]] = None,
                             model_dir: str = "./",
                             gamma: float = 0.99,
                             tau: float = 0.01,
                             delay: int = 3):
                    self.tau = tau
                    self.gamma = gamma
                    self.delay = delay
                    super(TD3_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

                def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                    self.iterations += 1
                    act_batch = torch.as_tensor(act_batch, device=self.device)
                    rew_batch = torch.as_tensor(rew_batch, device=self.device).unsqueeze(dim=1)
                    ter_batch = torch.as_tensor(terminal_batch, device=self.device).unsqueeze(dim=1)

                    # critic update
                    _, action_q = self.policy.Qaction(obs_batch, act_batch)
                    _, target_q = self.policy.Qtarget(next_batch)
                    backup = rew_batch + self.gamma * (1 - ter_batch) * target_q
                    q_loss = F.mse_loss(torch.tile(backup.detach(), (1, 2)), action_q)
                    self.optimizer[1].zero_grad()
                    q_loss.backward()
                    self.optimizer[1].step()
                    if self.scheduler is not None:
                        self.scheduler[1].step()

                    # actor update
                    if self.iterations % self.delay == 0:
                        _, policy_q = self.policy.Qpolicy(obs_batch)
                        p_loss = -policy_q.mean()
                        self.optimizer[0].zero_grad()
                        p_loss.backward()
                        self.optimizer[0].step()
                        if self.scheduler is not None:
                            self.scheduler[0].step()
                        self.policy.soft_update(self.tau)

                    actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
                    critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

                    info = {
                        "Qloss": q_loss.item(),
                        "Qvalue": action_q.mean().item(),
                        "actor_lr": actor_lr,
                        "critic_lr": critic_lr
                    }
                    if self.iterations % self.delay == 0:
                        info["Ploss"] = p_loss.item()

                    return info






  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        # TD3 add three tricks to DDPG:
        # 1. noisy action in target actor
        # 2. double critic network
        # 3. delayed actor update
        from xuance.mindspore.learners import *


        class TD3_Learner(Learner):
            class ActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(TD3_Learner.ActorNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._mean = ms.ops.ReduceMean(keep_dims=True)

                def construct(self, x):
                    _, policy_q = self._backbone.Qpolicy(x)
                    loss_p = -self._mean(policy_q)
                    return loss_p

            class CriticNetWithLossCell(nn.Cell):
                def __init__(self, backbone, gamma):
                    super(TD3_Learner.CriticNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._loss = nn.MSELoss()
                    self._gamma = gamma

                def construct(self, x, a, x_, r, d):
                    _, action_q = self._backbone.Qaction(x, a)
                    _, target_q = self._backbone.Qtarget(x_)
                    backup = r + self._gamma * (1 - d) * target_q
                    loss_q = self._loss(logits=action_q, labels=backup)
                    return loss_q

            def __init__(self,
                         policy: nn.Cell,
                         optimizers: nn.Optimizer,
                         schedulers: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01,
                         delay: int = 3):
                self.tau = tau
                self.gamma = gamma
                self.delay = delay
                super(TD3_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
                self._expand_dims = ms.ops.ExpandDims()
                # define mindspore trainers
                self.actor_loss_net = self.ActorNetWithLossCell(policy)
                self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
                self.actor_train.set_train()
                self.critic_loss_net = self.CriticNetWithLossCell(policy, self.gamma)
                self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
                self.critic_train.set_train()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                info = {}
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                rew_batch = self._expand_dims(Tensor(rew_batch), 1)
                next_batch = Tensor(next_batch)
                ter_batch = self._expand_dims(Tensor(terminal_batch), 1)

                q_loss = self.critic_train(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

                # actor update
                if self.iterations % self.delay == 0:
                    p_loss = self.actor_train(obs_batch)
                    self.policy.soft_update(self.tau)
                    info["Ploss"] = p_loss.asnumpy()

                actor_lr = self.scheduler['actor'](self.iterations).asnumpy()
                critic_lr = self.scheduler['critic'](self.iterations).asnumpy()

                info.update({
                    "Qloss": q_loss.asnumpy(),
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr
                })

                return info
