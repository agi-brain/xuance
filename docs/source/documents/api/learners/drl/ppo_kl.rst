PPOKL_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.policy_gradient.ppokl_learner.PPOKL_Learner(policy, optimizer, scheduler, device, model_dir, vf_coef, ent_coef, target_kl)

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
  :param target_kl: Target KL divergence used in PPO.
  :type target_kl: float

.. py:function::
  xuance.torch.learners.policy_gradient.ppokl_learner.PPOKL_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old action distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.policy_gradient.ppokl_learner.PPOKL_Learner(policy, optimizer, device, model_dir, vf_coef, ent_coef, target_kl)

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
  :param target_kl: Target KL divergence used in PPO.
  :type target_kl: float

.. py:function::
  xuance.tensorflow.learners.policy_gradient.ppokl_learner.PPOKL_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old action distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.policy_gradient.ppokl_learner.PPOKL_Learner(policy, optimizer, scheduler, summary_writer, model_dir, vf_coef, ent_coef, clip_range)

  :param policy: The policy that provides actions and values.
  :type policy: nn.Module
  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param scheduler: The tool for learning rate decay.
  :type scheduler: lr_scheduler
  :param summary_writer: The summary writer.
  :param model_dir: The directory for saving or loading the model parameters.
  :type model_dir: str
  :param vf_coef: Value function coefficient.
  :type vf_coef: float
  :param ent_coef: Entropy coefficient.
  :type ent_coef: float
  :param clip_range: PPO clip range.
  :type clip_range: float

.. py:function::
  xuance.mindspore.learners.policy_gradient.ppokl_learner.PPOKL_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_logp)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_logp: The previous log of actions.
  :type old_logp: np.ndarray
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
        from xuance.torch.utils.operations import merge_distributions


        class PPOKL_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         target_kl: float = 0.25):
                super(PPOKL_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.target_kl = target_kl
                self.kl_coef = 1.0

            def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                self.iterations += 1
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                adv_batch = torch.as_tensor(adv_batch, device=self.device)

                _, a_dist, v_pred = self.policy(obs_batch)
                log_prob = a_dist.log_prob(act_batch)
                old_dist = merge_distributions(old_dists)
                kl = a_dist.kl_divergence(old_dist).mean()
                old_logp_batch = old_dist.log_prob(act_batch)

                # ppo-clip core implementations
                ratio = (log_prob - old_logp_batch).exp().float()
                a_loss = -(ratio * adv_batch).mean() + self.kl_coef * kl
                c_loss = F.mse_loss(v_pred, ret_batch)
                e_loss = a_dist.entropy().mean()
                loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                if kl > self.target_kl * 1.5:
                    self.kl_coef = self.kl_coef * 2.
                elif kl < self.target_kl * 0.5:
                    self.kl_coef = self.kl_coef / 2.
                self.kl_coef = np.clip(self.kl_coef, 0.1, 20)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "actor-loss": a_loss.item(),
                    "critic-loss": c_loss.item(),
                    "entropy": e_loss.item(),
                    "learning_rate": lr,
                    "kl": kl.item(),
                    "predict_value": v_pred.mean().item()
                }

                return info


  .. group-tab:: TensorFlow

    .. code-block:: python

        from torch import kl_div
        from xuance.tensorflow.learners import *
        from xuance.tensorflow.utils.operations import merge_distributions


        class PPOKL_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         target_kl: float = 0.25):
                super(PPOKL_Learner, self).__init__(policy, optimizer, device, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.target_kl = target_kl
                self.kl_coef = 1.0

            def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                self.iterations += 1
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch)
                    ret_batch = tf.convert_to_tensor(ret_batch)
                    adv_batch = tf.convert_to_tensor(adv_batch)

                    with tf.GradientTape() as tape:
                        outputs, _, v_pred = self.policy(obs_batch)
                        a_dist = self.policy.actor.dist
                        log_prob = a_dist.log_prob(act_batch)
                        old_dist = merge_distributions(old_dists)
                        kl = tf.reduce_mean(a_dist.kl_divergence(old_dist))
                        old_logp_batch = old_dist.log_prob(act_batch)

                        # ppo-clip core implementations
                        ratio = tf.math.exp(log_prob - old_logp_batch)
                        a_loss = -tf.reduce_mean(ratio * adv_batch) + self.kl_coef * kl
                        c_loss = tk.losses.mean_squared_error(ret_batch, v_pred)
                        e_loss = tf.reduce_mean(a_dist.entropy())
                        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                        if kl > self.target_kl * 1.5:
                            self.kl_coef = self.kl_coef * 2.
                        elif kl < self.target_kl * 0.5:
                            self.kl_coef = self.kl_coef / 2.
                        self.kl_coef = np.clip(self.kl_coef, 0.1, 20)
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
                        "kl": kl.numpy(),
                        "predict_value": tf.math.reduce_mean(v_pred).numpy()
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
                         summary_writer: Optional[SummaryWriter] = None,
                         model_dir: str = "./",
                         vf_coef: float = 0.25,
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25):
                super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, summary_writer, model_dir)
                self.vf_coef = vf_coef
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                # define mindspore trainer
                self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef, self.vf_coef, self.clip_range)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
                self.iterations += 1
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                ret_batch = Tensor(ret_batch)
                adv_batch = Tensor(adv_batch)
                old_logp_batch = Tensor(old_logp)

                loss = self.policy_train(obs_batch, act_batch, old_logp_batch, adv_batch, ret_batch)
                # Logger
                lr = self.scheduler(self.iterations).asnumpy()
                self.writer.add_scalar("tot-loss", loss.asnumpy(), self.iterations)
                self.writer.add_scalar("learning_rate", lr, self.iterations)
