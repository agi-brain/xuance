PPOCLIP_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner(policy, optimizer, scheduler, device, model_dir, vf_coef, ent_coef, clip_range, clip_grad_norm, use_grad_clip)

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
  :param vf_coef: Value function coefficient.
  :type vf_coef: float
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_range: PPO clip range.
  :type clip_range: float
  :param clip_grad_norm: the maximum norm of the gradients used for gradient clipping.
  :type clip_grad_norm: float
  :param use_grad_clip: gradient clipping should be applied during backpropagation.
  :type use_grad_clip: bool

.. py:function::
  xuance.torch.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch,old_logp)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param value_batch: A batch of predicted values by the value function.
  :type value_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_logp: A batch of old log probabilities of actions.
  :type old_logp: tensor
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner(policy, optimizer, device, model_dir, vf_coef, ent_coef, clip_range)

  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param device: The calculating device.
  :type device: str
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param vf_coef: Value function coefficient.
  :type vf_coef: float
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_range: PPO clip range.
  :type clip_range: float

.. py:function::
  xuance.tensorflow.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch,old_logp)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param value_batch: A batch of predicted values by the value function.
  :type value_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_logp: A batch of old log probabilities of actions.
  :type old_logp: tensor
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner(policy, optimizer, scheduler, model_dir, vf_coef, ent_coef, clip_range)

  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param vf_coef: Value function coefficient.
  :type vf_coef: float
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_range: PPO clip range.
  :type clip_range: float

.. py:function::
  xuance.mindspore.learners.policy_gradient.ppoclip_learner.PPOCLIP_Learner.update(obs_batch, act_batch, ret_batch, value_batch, adv_batch,old_logp)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param value_batch: A batch of predicted values by the value function.
  :type value_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_logp: A batch of old log probabilities of actions.
  :type old_logp: tensor
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


        class PPOCLIP_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25,
                         clip_grad_norm: float = 0.25,
                         use_grad_clip: bool = True,
                         ):
                super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                self.clip_grad_norm = clip_grad_norm
                self.use_grad_clip = use_grad_clip

            def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                value_batch = torch.as_tensor(value_batch, device=self.device)
                adv_batch = torch.as_tensor(adv_batch, device=self.device)
                old_logp_batch = torch.as_tensor(old_logp, device=self.device)

                outputs, a_dist, v_pred = self.policy(obs_batch)
                log_prob = a_dist.log_prob(act_batch)

                # ppo-clip core implementations
                ratio = (log_prob - old_logp_batch).exp().float()
                surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                surrogate2 = adv_batch * ratio
                a_loss = -torch.minimum(surrogate1, surrogate2).mean()

                c_loss = F.mse_loss(v_pred, ret_batch)

                e_loss = a_dist.entropy().mean()
                loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]

                info = {
                    "actor-loss": a_loss.item(),
                    "critic-loss": c_loss.item(),
                    "entropy": e_loss.item(),
                    "learning_rate": lr,
                    "predict_value": v_pred.mean().item(),
                    "clip_ratio": cr
                }

                return info




  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.learners import *


        class PPOCLIP_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25):
                super(PPOCLIP_Learner, self).__init__(policy, optimizer, device, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.clip_range = clip_range

            def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
                self.iterations += 1
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch)
                    ret_batch = tf.convert_to_tensor(ret_batch)
                    adv_batch = tf.convert_to_tensor(adv_batch)
                    old_logp_batch = tf.convert_to_tensor(old_logp)

                    with tf.GradientTape() as tape:
                        outputs, a_dist, v_pred = self.policy(obs_batch)
                        a_dist = self.policy.actor.dist
                        log_prob = a_dist.log_prob(act_batch)

                        # ppo-clip core implementations
                        ratio = tf.math.exp(log_prob - old_logp_batch)
                        surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                        surrogate2 = adv_batch * ratio
                        a_loss = -tf.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
                        c_loss = tk.losses.mean_squared_error(ret_batch, v_pred)
                        e_loss = tf.reduce_mean(a_dist.entropy())
                        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])

                    lr = self.optimizer._decayed_lr(tf.float32)
                    info = {
                        "actor-loss": a_loss.numpy(),
                        "critic-loss": c_loss.numpy(),
                        "entropy": e_loss.numpy(),
                        "learning_rate": lr.numpy(),
                        "predict_value": tf.math.reduce_mean(v_pred).numpy(),
                    }

                    return info


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *


        class PPOCLIP_Learner(Learner):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, ent_coef, vf_coef, clip_range):
                    super(PPOCLIP_Learner.PolicyNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self._ent_coef = ent_coef
                    self._vf_coef = vf_coef
                    self._clip_range = [Tensor(1.0 - clip_range), Tensor(1.0 + clip_range)]
                    self._exp = ms.ops.Exp()
                    self._minimum = ms.ops.Minimum()
                    self._mean = ms.ops.ReduceMean(keep_dims=True)
                    self._loss = nn.MSELoss()

                def construct(self, x, a, old_log_p, adv, ret):
                    outputs, act_probs, v_pred = self._backbone(x)
                    log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs)
                    ratio = self._exp(log_prob - old_log_p)
                    surrogate1 = ms.ops.clip_by_value(ratio, self._clip_range[0], self._clip_range[1]) * adv
                    surrogate2 = adv * ratio
                    loss_a = -self._mean(self._minimum(surrogate1, surrogate2))
                    loss_c = self._loss(logits=v_pred, labels=ret)
                    loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
                    loss = loss_a - self._ent_coef * loss_e + self._vf_coef * loss_c
                    return loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25):
                super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                # define mindspore trainer
                self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef, self.vf_coef, self.clip_range)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                ret_batch = Tensor(ret_batch)
                value_batch = Tensor(value_batch)
                adv_batch = Tensor(adv_batch)
                old_logp_batch = Tensor(old_logp)

                loss = self.policy_train(obs_batch, act_batch, old_logp_batch, adv_batch, ret_batch)
                # Logger
                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "tot-loss": loss.asnumpy(),
                    "learning_rate": lr
                }
                return info
