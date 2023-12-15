PG_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.pg_learner.PG_Learner(policy, optimizer, scheduler, device, model_dir, ent_coef, clip_grad)

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
  :param ent_coef: xxxxxx.
  :type ent_coef: xxxxxx
  :param clip_grad: xxxxxx.
  :type clip_grad: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.pg_learner.PG_Learner.update(obs_batch, act_batch, ret_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.learners.policy_gradient.pg_learner.PG_Learner(policy, optimizer, scheduler, model_dir, ent_coef, clip_grad, clip_type)

  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param ent_coef: xxxxxx.
  :type ent_coef: xxxxxx
  :param clip_grad: xxxxxx.
  :type clip_grad: xxxxxx
  :param clip_type: xxxxxx.
  :type clip_type: xxxxxx

.. py:function::
  xuance.mindspore.learners.policy_gradient.pg_learner.PG_Learner.update(obs_batch, act_batch, ret_batch)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
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


        class PG_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_grad: Optional[float] = None):
                super(PG_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
                self.ent_coef = ent_coef
                self.clip_grad = clip_grad

            def update(self, obs_batch, act_batch, ret_batch):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                _, a_dist = self.policy(obs_batch)
                log_prob = a_dist.log_prob(act_batch)

                a_loss = -(ret_batch * log_prob).mean()
                e_loss = a_dist.entropy().mean()

                loss = a_loss - self.ent_coef * e_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "actor-loss": a_loss.item(),
                    "entropy": e_loss.item(),
                    "learning_rate": lr
                }

                return info


  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *


        class PG_Learner(Learner):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, ent_coef):
                    super(PG_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self._ent_coef = ent_coef
                    self._mean = ms.ops.ReduceMean(keep_dims=True)

                def construct(self, x, a, r):
                    _, act_probs = self._backbone(x)
                    log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs)
                    loss_a = -self._mean(r * log_prob)
                    loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
                    loss = loss_a - self._ent_coef * loss_e
                    return loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_grad: Optional[float] = None,
                         clip_type: Optional[int] = None):
                super(PG_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                self.ent_coef = ent_coef
                self.clip_grad = clip_grad
                # define mindspore trainer
                self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef)
                # self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, optimizer,
                                                                 clip_type=clip_type, clip_value=clip_grad)
                self.policy_train.set_train()

            def update(self, obs_batch, act_batch, ret_batch):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                ret_batch = Tensor(ret_batch)

                loss = self.policy_train(obs_batch, act_batch, ret_batch)

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "total-loss": loss.asnumpy(),
                    "learning_rate": lr
                }

                return info
