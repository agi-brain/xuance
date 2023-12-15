SPDQN_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.spdqn_learner.SPDQN_Learner(policy, optimizer, scheduler, device, model_dir, gamma, tau)

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
  xuance.torch.learners.policy_gradient.spdqn_learner.SPDQN_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

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
  xuance.mindspore.learners.policy_gradient.spdqn_learner.SPDQN_Learner(policy, optimizer, scheduler, model_dir, gamma, tau)

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
  xuance.mindspore.learners.policy_gradient.spdqn_learner.SPDQN_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

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


        class SPDQN_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizers: Sequence[torch.optim.Optimizer],
                         schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                         summary_writer: Optional[SummaryWriter] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01):
                self.tau = tau
                self.gamma = gamma
                super(SPDQN_Learner, self).__init__(policy, optimizers, schedulers, summary_writer, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = torch.as_tensor(obs_batch, device=self.device)
                hyact_batch = torch.as_tensor(act_batch, device=self.device)
                disact_batch = hyact_batch[:, 0].long()
                conact_batch = hyact_batch[:, 1:]
                rew_batch = torch.as_tensor(rew_batch, device=self.device)
                next_batch = torch.as_tensor(next_batch, device=self.device)
                ter_batch = torch.as_tensor(terminal_batch, device=self.device)

                # optimize Q-network
                with torch.no_grad():
                    target_conact = self.policy.Atarget(next_batch)
                    target_q = self.policy.Qtarget(next_batch, target_conact)
                    target_q = torch.max(target_q, 1, keepdim=True)[0].squeeze()

                    target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

                eval_qs = self.policy.Qeval(obs_batch, conact_batch)
                eval_q = eval_qs.gather(1, disact_batch.view(-1, 1)).squeeze()
                q_loss = F.mse_loss(eval_q, target_q)

                self.optimizer[1].zero_grad()
                q_loss.backward()
                self.optimizer[1].step()

                # optimize actor network
                policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = - policy_q.mean()
                self.optimizer[0].zero_grad()
                p_loss.backward()
                self.optimizer[0].step()

                if self.scheduler is not None:
                    self.scheduler[0].step()
                    self.scheduler[1].step()

                self.policy.soft_update(self.tau)

                self.writer.add_scalar("Q_loss", q_loss.item(), self.iterations)
                self.writer.add_scalar("P_loss", q_loss.item(), self.iterations)
                self.writer.add_scalar('Qvalue', eval_q.mean().item(), self.iterations)








  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *
        from mindspore.ops import OneHot


        class SPDQN_Learner(Learner):
            class QNetWithLossCell(nn.Cell):
                def __init__(self, backbone, loss_fn):
                    super(SPDQN_Learner.QNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self._loss_fn = loss_fn

                def construct(self, x, dis_a, con_a, label, input_q):
                    # optimize q-network
                    eval_qs = self._backbone.Qeval(x, con_a, input_q)
                    eval_q = eval_qs.gather(dis_a.astype(ms.int32).view(-1, 1), 1).squeeze()
                    q_loss = self._loss_fn(eval_q, label)
                    return q_loss

            class ConActorNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(SPDQN_Learner.ConActorNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone

                def construct(self, x, input_q2):
                    # optimize actor network
                    policy_q = self._backbone.Qpolicy(x, input_q2)
                    p_loss = - policy_q.mean()
                    return p_loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: Sequence[nn.Optimizer],
                         scheduler: Optional[Sequence[nn.exponential_decay_lr]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         tau: float = 0.01):
                self.gamma = gamma
                self.tau = tau
                super(SPDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                # define loss function
                loss_fn = nn.MSELoss()
                # connect the feed forward network with loss function.
                self.q_loss_net = self.QNetWithLossCell(policy, loss_fn)
                self.con_loss_net = self.ConActorNetWithLossCell(policy)
                # define the training network
                self.con_actor_train = nn.TrainOneStepCell(self.con_loss_net, optimizer[0])
                self.q_net_train = nn.TrainOneStepCell(self.q_loss_net, optimizer[1])
                # set the training network as train mode.
                self.con_actor_train.set_train()
                self.q_net_train.set_train()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                hyact_batch = Tensor(act_batch)
                disact_batch = hyact_batch[:, 0]  # .long()
                conact_batch = hyact_batch[:, 1:]
                rew_batch = Tensor(rew_batch)
                next_batch = Tensor(next_batch)
                ter_batch = Tensor(terminal_batch)

                target_conact = self.policy.Atarget(next_batch)
                target_q = self.policy.Qtarget(next_batch, target_conact)
                target_q = target_q.max(axis=-1)
                target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

                batch_size = obs_batch.shape[0]
                input_q = self.policy._concat((obs_batch, self.policy._zeroslike(conact_batch)))
                input_q = input_q.repeat(self.policy.num_disact, 0)
                input_q = input_q.asnumpy()
                conact_batch = conact_batch.asnumpy()
                for i in range(self.policy.num_disact):
                    input_q[i * batch_size:(i + 1) * batch_size,
                    self.policy.obs_size + self.policy.offsets[i]: self.policy.obs_size + self.policy.offsets[i + 1]] \
                        = conact_batch[:, self.policy.offsets[i]:self.policy.offsets[i + 1]]
                input_q = ms.Tensor(input_q, dtype=ms.float32)
                conact_batch = Tensor(conact_batch)

                conact = self.policy.conactor(obs_batch)
                input_q2 = self.policy._concat((obs_batch, self.policy._zeroslike(conact)))
                input_q2 = input_q2.repeat(self.policy.num_disact, 0)
                input_q2 = input_q2.asnumpy()
                conact = conact.asnumpy()
                for i in range(self.policy.num_disact):
                    input_q2[i * batch_size:(i + 1) * batch_size,
                    self.policy.obs_size + self.policy.offsets[i]: self.policy.obs_size + self.policy.offsets[i + 1]] \
                        = conact[:, self.policy.offsets[i]:self.policy.offsets[i + 1]]
                input_q2 = ms.Tensor(input_q2, dtype=ms.float32)

                q_loss = self.q_net_train(obs_batch, disact_batch, conact_batch, target_q, input_q)
                p_loss = self.con_actor_train(obs_batch, input_q2)

                self.policy.soft_update(self.tau)

                con_actor_lr = self.scheduler[0](self.iterations).asnumpy()
                qnet_lr = self.scheduler[1](self.iterations).asnumpy()

                info = {
                    "P_loss": p_loss.asnumpy(),
                    "Q_loss": q_loss.asnumpy(),
                    "con_actor_lr": con_actor_lr,
                    "qnet_lr": qnet_lr
                }

                return info
