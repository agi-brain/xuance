PPOKL_Learner
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.policy_gradient.ppokl_learner.PPOKL_Learner(policy, optimizer, scheduler, device, model_dir, vf_coef, ent_coef, target_kl)

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
  :param vf_coef: xxxxxx.
  :type vf_coef: xxxxxx
  :param ent_coef: xxxxxx.
  :type ent_coef: xxxxxx
  :param target_kl: xxxxxx.
  :type target_kl: xxxxxx

.. py:function::
  xuance.torch.learners.policy_gradient.ppokl_learner.PPOKL_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

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
  xuance.mindspore.learners.policy_gradient.ppokl_learner.PPOKL_Learner(policy, optimizer, scheduler, summary_writer, model_dir, vf_coef, ent_coef, clip_range)

  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param summary_writer: xxxxxx.
  :type summary_writer: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param vf_coef: xxxxxx.
  :type vf_coef: xxxxxx
  :param ent_coef: xxxxxx.
  :type ent_coef: xxxxxx
  :param clip_range: xxxxxx.
  :type clip_range: xxxxxx

.. py:function::
  xuance.mindspore.learners.policy_gradient.ppokl_learner.PPOKL_Learner.update(obs_batch, act_batch, ret_batch, adv_batch, old_logp)

  :param obs_batch: xxxxxx.
  :type obs_batch: xxxxxx
  :param act_batch: xxxxxx.
  :type act_batch: xxxxxx
  :param ret_batch: xxxxxx.
  :type ret_batch: xxxxxx
  :param adv_batch: xxxxxx.
  :type adv_batch: xxxxxx
  :param old_logp: xxxxxx.
  :type old_logp: xxxxxx
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
