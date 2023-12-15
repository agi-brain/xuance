PPG_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner(policy, optimizer, scheduler, device, model_dir, ent_coef, clip_range, kl_beta)

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
  :param clip_range: xxxxxx.
  :type clip_range: xxxxxx
  :param kl_beta: xxxxxx.
  :type kl_beta: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_policy(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :param old_dists: xxxxxx.
  :type old_dists: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_critic(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :param old_dists: xxxxxx.
  :type old_dists: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.ppg_learner.PPG_Learner.update_auxiliary(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :param old_dists: xxxxxx.
  :type old_dists: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.learners.policy_gradient.ppg_learner.PPG_Learner(policy, optimizer, scheduler, model_dir, ent_coef, clip_range, kl_beta)

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
  :param clip_range: xxxxxx.
  :type clip_range: xxxxxx
  :param kl_beta: xxxxxx.
  :type kl_beta: xxxxxx

.. py:function::
  xuance.mindspore.learners.policy_gradient.ppg_learner.PPG_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_dists, update_type)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :param old_dists: xxxxxx.
  :type old_dists: xxxxxx
  :param update_type: xxxxxx.
  :type update_type: xxxxxx
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
