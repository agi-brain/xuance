QRDQN_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.qlearning_family.qrdqn_learner.QRDQN_Learner(policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.torch.learners.qlearning_family.qrdqn_learner.QRDQN_Learner.update(obs_batch, act_batch, rew_batch, terminal_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param rew_batch: xxxxxx.
  :type rew_batch: xxxxxx
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
  xuance.mindspore.learners.qlearning_family.qrdqn_learner.QRDQN_Learner(policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

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
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.mindspore.learners.qlearning_family.qrdqn_learner.QRDQN_Learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

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


        class DRQN_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(DRQN_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

            def update(self, obs_batch, act_batch, rew_batch, terminal_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                rew_batch = torch.as_tensor(rew_batch, device=self.device)
                ter_batch = torch.as_tensor(terminal_batch, device=self.device, dtype=torch.float)
                batch_size = obs_batch.shape[0]

                rnn_hidden = self.policy.init_hidden(batch_size)
                _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
                target_rnn_hidden = self.policy.init_hidden(batch_size)
                _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
                # targetQ = targetQ.max(dim=-1).values

                targetA = F.one_hot(targetA, targetQ.shape[-1])
                targetQ = (targetQ * targetA).sum(dim=-1)

                targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
                predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[-1])).sum(dim=-1)

                loss = F.mse_loss(predictQ, targetQ)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # hard update for target network
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "Qloss": loss.item(),
                    "learning_rate": lr,
                    "predictQ": predictQ.mean().item()
                }

                return info



  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *
        from mindspore.ops import OneHot,ExpandDims,ReduceSum


        class QRDQN_Learner(Learner):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, loss_fn):
                    super(QRDQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self._loss_fn = loss_fn
                    self._onehot = OneHot()
                    self.on_value = Tensor(1.0, ms.float32)
                    self.off_value = Tensor(0.0, ms.float32)
                    self._unsqueeze = ExpandDims()
                    self._sum = ReduceSum()

                def construct(self, x, a, target_quantile):
                    _,_,evalZ = self._backbone(x)
                    current_quantile = self._sum(evalZ * self._unsqueeze(self._onehot(a, evalZ.shape[1], self.on_value, self.off_value), -1), 1)
                    loss = self._loss_fn(target_quantile, current_quantile)
                    return loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                super(QRDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                # define loss function
                loss_fn = nn.MSELoss()
                # connect the feed forward network with loss function.
                self.loss_net = self.PolicyNetWithLossCell(policy, loss_fn)
                # define the training network
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                # set the training network as train mode.
                self.policy_train.set_train()

                self._onehot = OneHot()
                self.on_value = Tensor(1.0, ms.float32)
                self.off_value = Tensor(0.0, ms.float32)
                self._unsqueeze = ExpandDims()
                self._sum = ReduceSum()

            def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch, ms.int32)
                rew_batch = Tensor(rew_batch)
                next_batch = Tensor(next_batch)
                ter_batch = Tensor(terminal_batch)

                _, targetA, targetZ = self.policy(next_batch)
                target_quantile = self._sum(targetZ * self._unsqueeze(self._onehot(targetA, targetZ.shape[1], self.on_value, self.off_value), -1), 1)
                target_quantile = self._unsqueeze(rew_batch, 1) + self.gamma * target_quantile * (1-self._unsqueeze(ter_batch, 1))

                loss = self.policy_train(obs_batch, act_batch, target_quantile)

                # hard update for target network
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "Qloss": loss.asnumpy(),
                    "learning_rate": lr
                }

                return info

