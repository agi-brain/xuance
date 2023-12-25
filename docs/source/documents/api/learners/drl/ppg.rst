PPG_Learner
=====================================

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner(policy, optimizer, scheduler, device, model_dir, ent_coef, clip_range, kl_beta)

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
  :param clip_range: PPO clip range.
  :type clip_range: float
  :param kl_beta: KL divergence coefficient.
  :type kl_beta: float

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_policy(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_critic(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_auxiliary(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.learners.policy_gradient.ppg_learner.PPG_Learner(policy, optimizer, device, model_dir, ent_coef, clip_range, kl_beta)

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
  :param clip_range: PPO clip range.
  :type clip_range: float
  :param kl_beta: KL divergence coefficient.
  :type kl_beta: float

.. py:function::
  xuance.tensorflow.learners.policy_gradient.ppg_learner.PPG_Learner.update_policy(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.tensorflow.learners.policy_gradient.ppg_learner.PPG_Learner.update_critic(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. py:function::
  xuance.tensorflow.learners.policy_gradient.ppg_learner.PPG_Learner.update_auxiliary(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: Batch of old distributions.
  :type old_dists: list
  :return: The information of the training.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.learners.policy_gradient.ppg_learner.PPG_Learner(policy, optimizer, scheduler, model_dir, ent_coef, clip_range, kl_beta)

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
  :param clip_range: PPO clip range.
  :type clip_range: float
  :param kl_beta: KL divergence coefficient.
  :type kl_beta: float

.. py:function::
  xuance.mindspore.learners.policy_gradient.ppg_learner.PPG_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_dists, update_type)

  :param obs_batch: A batch of observations sampled from experience replay buffer.
  :type obs_batch: np.ndarray
  :param act_batch: A batch of actions sampled from experience replay buffer.
  :type act_batch: np.ndarray
  :param ret_batch: A batch of returns sampled from experience replay buffer.
  :type ret_batch: np.ndarray
  :param adv_batch: A batch of advantages sampled from experience replay buffer.
  :type adv_batch: np.ndarray
  :param old_dists: old distributions.
  :type old_dists: list
  :param update_type: int.
  :type update_type: the type of update (0 for actor, 1 for critic, 2 for auxiliary)
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


        class PPG_Learner(Learner):
            def __init__(self,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25,
                         kl_beta: float = 1.0):
                super(PPG_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                self.kl_beta = kl_beta
                self.policy_iterations = 0
                self.value_iterations = 0

            def update_policy(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                adv_batch = torch.as_tensor(adv_batch, device=self.device)
                old_dist = merge_distributions(old_dists)
                old_logp_batch = old_dist.log_prob(act_batch).detach()

                outputs, a_dist, _, _ = self.policy(obs_batch)
                log_prob = a_dist.log_prob(act_batch)
                # ppo-clip core implementations
                ratio = (log_prob - old_logp_batch).exp().float()
                surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                surrogate2 = adv_batch * ratio
                a_loss = -torch.minimum(surrogate1, surrogate2).mean()
                e_loss = a_dist.entropy().mean()
                loss = a_loss - self.ent_coef * e_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # Logger
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]

                info = {
                    "actor-loss": a_loss.item(),
                    "entropy": e_loss.item(),
                    "learning_rate": lr,
                    "clip_ratio": cr,
                }
                self.policy_iterations += 1

                return info

            def update_critic(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                _, _, v_pred, _ = self.policy(obs_batch)
                loss = F.mse_loss(v_pred, ret_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                info = {
                    "critic-loss": loss.item()
                }
                self.value_iterations += 1
                return info

            def update_auxiliary(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                act_batch = torch.as_tensor(act_batch, device=self.device)
                ret_batch = torch.as_tensor(ret_batch, device=self.device)
                adv_batch = torch.as_tensor(adv_batch, device=self.device)

                old_dist = merge_distributions(old_dists)
                outputs, a_dist, v, aux_v = self.policy(obs_batch)
                aux_loss = F.mse_loss(v.detach(), aux_v)
                kl_loss = a_dist.kl_divergence(old_dist).mean()
                value_loss = F.mse_loss(v, ret_batch)
                loss = aux_loss + self.kl_beta * kl_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                info = {
                    "kl-loss": loss.item()
                }
                return info

            def update(self):
                pass



  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.learners import *
        from xuance.tensorflow.utils.operations import merge_distributions


        class PPG_Learner(Learner):
            def __init__(self,
                         policy: tk.Model,
                         optimizer: tk.optimizers.Optimizer,
                         device: str = "cpu:0",
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25,
                         kl_beta: float = 1.0):
                super(PPG_Learner, self).__init__(policy, optimizer, device, model_dir)
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                self.kl_beta = kl_beta
                self.policy_iterations = 0
                self.value_iterations = 0

            def update_policy(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch)
                    ret_batch = tf.convert_to_tensor(ret_batch)
                    adv_batch = tf.convert_to_tensor(adv_batch)

                    with tf.GradientTape() as tape:
                        old_dist = merge_distributions(old_dists)
                        old_logp_batch = tf.stop_gradient(old_dist.log_prob(act_batch))

                        outputs, _, _, _ = self.policy(obs_batch)
                        a_dist = self.policy.actor.dist
                        log_prob = a_dist.log_prob(act_batch)
                        # ppo-clip core implementations
                        ratio = tf.math.exp(log_prob - old_logp_batch)
                        surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                        surrogate2 = adv_batch * ratio

                        a_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                        e_loss = tf.reduce_mean(a_dist.entropy())
                        loss = a_loss - self.ent_coef * e_loss
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])
                    lr_policy = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "actor-loss": a_loss.numpy(),
                        "entropy": e_loss.numpy(),
                        "learning_rate": lr_policy.numpy(),
                    }
                    self.policy_iterations += 1

                    return info

            def update_critic(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                with tf.device(self.device):
                    ret_batch = tf.convert_to_tensor(ret_batch)
                    with tf.GradientTape() as tape:
                        _, _, v_pred, _ = self.policy(obs_batch)
                        loss = tk.losses.mean_squared_error(ret_batch, v_pred)
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])
                    lr_critic = self.optimizer._decayed_lr(tf.float32)
                    info = {
                        "critic-loss": loss.numpy(),
                        "lr_critic": lr_critic.numpy()
                    }
                    self.value_iterations += 1
                    return info

            def update_auxiliary(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
                with tf.device(self.device):
                    act_batch = tf.convert_to_tensor(act_batch)
                    ret_batch = tf.convert_to_tensor(ret_batch)
                    adv_batch = tf.convert_to_tensor(adv_batch)

                    with tf.GradientTape() as tape:
                        old_dist = merge_distributions(old_dists)
                        outputs, _, v, aux_v = self.policy(obs_batch)
                        a_dist = self.policy.actor.dist
                        aux_loss = tk.losses.mean_squared_error(tf.stop_gradient(v), aux_v)
                        kl_loss = tf.reduce_mean(a_dist.kl_divergence(old_dist))
                        value_loss = tk.losses.mean_squared_error(ret_batch, v)
                        loss = aux_loss + self.kl_beta * kl_loss + value_loss
                        gradients = tape.gradient(loss, self.policy.trainable_variables)
                        self.optimizer.apply_gradients([
                            (grad, var)
                            for (grad, var) in zip(gradients, self.policy.trainable_variables)
                            if grad is not None
                        ])
                    lr_aux = self.optimizer._decayed_lr(tf.float32)

                    info = {
                        "kl-loss": loss.numpy(),
                        "lr_aux": lr_aux.numpy()
                    }
                    return info

            def update(self):
                pass


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.learners import *
        from xuance.mindspore.utils.operations import merge_distributions
        from mindspore.nn.probability.distribution import Categorical

        class PPG_Learner(Learner):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, ent_coef, kl_beta, clip_range, loss_fn):
                    super(PPG_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone
                    self._ent_coef = ent_coef
                    self._kl_beta = kl_beta
                    self._clip_range = clip_range
                    self._loss_fn = loss_fn
                    self._mean = ms.ops.ReduceMean(keep_dims=True)
                    self._minimum = ms.ops.Minimum()
                    self._exp = ms.ops.Exp()
                    self._categorical = Categorical()

                def construct(self, x, a, r, adv, old_log, old_dist_logits, v, update_type):
                    loss = 0
                    if update_type == 0:
                        _, a_dist, _, _ = self._backbone(x)
                        log_prob = self._categorical.log_prob(a, a_dist)
                        # ppo-clip core implementations
                        ratio = self._exp(log_prob - old_log)
                        surrogate1 = ms.ops.clip_by_value(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range) * adv
                        surrogate2 = adv * ratio
                        a_loss = -self._minimum(surrogate1, surrogate2).mean()
                        entropy = self._categorical.entropy(a_dist)
                        e_loss = entropy.mean()
                        loss = a_loss - self._ent_coef * e_loss
                    elif update_type == 1:
                        _,_,v_pred,_ = self._backbone(x)
                        loss = self._loss_fn(v_pred, r)
                    elif update_type == 2:
                        _, a_dist, _, aux_v  = self._backbone(x)
                        aux_loss = self._loss_fn(v, aux_v)
                        kl_loss = self._categorical.kl_loss('Categorical',a_dist, old_dist_logits).mean()
                        value_loss = self._loss_fn(v,r)
                        loss = aux_loss + self._kl_beta * kl_loss + value_loss
                    return loss

            def __init__(self,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         ent_coef: float = 0.005,
                         clip_range: float = 0.25,
                         kl_beta: float = 1.0):
                super(PPG_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
                self.ent_coef = ent_coef
                self.clip_range = clip_range
                self.kl_beta = kl_beta
                self.policy_iterations = 0
                self.value_iterations = 0
                loss_fn = nn.MSELoss()
                # define mindspore trainer
                self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef, self.kl_beta, self.clip_range, loss_fn)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists, update_type):
                self.iterations += 1
                info = {}
                obs_batch = Tensor(obs_batch)
                act_batch = Tensor(act_batch)
                ret_batch = Tensor(ret_batch)
                adv_batch = Tensor(adv_batch)
                old_dist = merge_distributions(old_dists)
                old_logp_batch = old_dist.log_prob(act_batch)

                _, _, v, _  = self.policy(obs_batch)

                if update_type == 0:
                    loss = self.policy_train(obs_batch, act_batch, ret_batch, adv_batch, old_logp_batch, old_dist.logits, v, update_type)

                    lr = self.scheduler(self.iterations).asnumpy()
                    # self.writer.add_scalar("actor-loss", self.loss_net.loss_a.asnumpy(), self.iterations)
                    # self.writer.add_scalar("entropy", self.loss_net.loss_e.asnumpy(), self.iterations)
                    info["total-loss"] = loss.asnumpy()
                    info["learning_rate"] = lr
                    self.policy_iterations += 1

                elif update_type == 1:
                    loss = self.policy_train(obs_batch, act_batch, ret_batch, adv_batch, old_logp_batch, old_dist.logits, v, update_type)

                    info["critic-loss"] = loss.asnumpy()
                    self.value_iterations += 1

                elif update_type == 2:
                    loss = self.policy_train(obs_batch, act_batch, ret_batch, adv_batch, old_logp_batch, old_dist.logits, v, update_type)

                    info["kl-loss"] = loss.asnumpy()

                return info
