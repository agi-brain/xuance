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