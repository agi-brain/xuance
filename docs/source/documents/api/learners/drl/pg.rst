PG_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.policy_gradient.pg_learner.PG_Learner(policy, optimizer, scheduler, device, model_dir, ent_coef, clip_grad)

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
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_grad: Gradient clipping threshold.
  :type clip_grad: float

.. py:function::
  xuance.torch.learners.policy_gradient.pg_learner.PG_Learner.update(obs_batch, act_batch, ret_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.policy_gradient.pg_learner.PG_Learner(policy, optimizer, device, model_dir, ent_coef, clip_grad)

  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_grad: Gradient clipping threshold.
  :type clip_grad: float

.. py:function::
  xuance.tensorflow.learners.policy_gradient.pg_learner.PG_Learner.update(obs_batch, act_batch, ret_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.policy_gradient.pg_learner.PG_Learner(policy, optimizer, scheduler, model_dir, ent_coef, clip_grad, clip_type)

  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_grad: Gradient clipping threshold.
  :type clip_grad: float
  :param clip_type: Type of gradient clipping.

.. py:function::
  xuance.mindspore.learners.policy_gradient.pg_learner.PG_Learner.update(obs_batch, act_batch, ret_batch)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
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

        from xuance.tensorflow.learners import *


        class PG_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_grad: Optional[float] = None):
                super(PG_Learner, self).__init__(policy, optimizer, device, model_dir)
                self.ent_coef = ent_coef
                self.clip_grad = clip_grad

            def update(self, obs_batch, act_batch, ret_batch):
                self.iterations += 1
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch, dtype=tf.float32)
                    ret_batch = tf.convert_to_tensor(ret_batch)

                    with tf.GradientTape() as tape:
                        outputs, _ = self.policy(obs_batch)
                        a_dist = self.policy.actor.dist
                        log_prob = a_dist.log_prob(act_batch)

                        a_loss = -tf.reduce_mean(ret_batch * log_prob)
                        e_loss = tf.reduce_mean(a_dist.entropy())

                        loss = a_loss - self.ent_coef * e_loss
                        gradients = tape.gradient(loss, self.policy.trainable_variables)

                        self.optimizer.apply_gradients([
                            (tf.clip_by_norm(grad, self.clip_grad), var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])

                    lr = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "actor-loss": a_loss.numpy(),
                        "entropy": e_loss.numpy(),
                        "learning_rate": lr.numpy()
                    }

                    return info


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
