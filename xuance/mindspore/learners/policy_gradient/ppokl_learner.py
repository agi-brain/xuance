"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads
from xuance.mindspore.utils.operations import merge_distributions
from argparse import Namespace
from mindspore.nn import MSELoss


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPOKL_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.target_kl = config.target_kl
        self.kl_coef = config.kl_coef
        self.clip_range = config.clip_range

        self._clip_range = [Tensor(1.0 - self.clip_range), Tensor(1.0 + self.clip_range)]
        self._exp = ms.ops.Exp()
        self._minimum = ms.ops.Minimum()
        self._mean = ms.ops.ReduceMean(keep_dims=True)
        self._loss = MSELoss()
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()
        self.continuous_control = True if ("gaussian" in str(type(self.policy))) else False

    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists_param):
        old_log_prob = old_dists_param['log_prob']
        outputs, act_dist, v_pred = self.policy(obs_batch)
        log_prob = act_dist.log_prob(act_batch)
        ratio = self._exp(log_prob - old_log_prob)
        if self.continuous_control:
            mean, std = old_dists_param['mean'], old_dists_param['std']
            kl = act_dist.distribution.kl_loss('Normal', mean, std).mean()
        else:
            probs = old_dists_param['probs']
            kl = act_dist.distribution.kl_loss('Categorical', probs).mean()
        loss_a = -(ratio * adv_batch).mean() + self.kl_coef * kl
        loss_c = self._loss(logits=v_pred, labels=ret_batch)
        loss_e = self._mean(act_dist.entropy())
        loss = loss_a - self.ent_coef * loss_e + self.vf_coef * loss_c
        return loss, loss_a, loss_c, loss_e, kl, v_pred

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_logp_batch = old_dist.log_prob(act_batch)
        if self.continuous_control:
            old_dists_param = {
                "mean": old_dist.mu,
                "std": old_dist.std,
                "log_prob": old_logp_batch
            }
        else:
            old_dists_param = {
                "probs": old_dist.probs,
                "log_prob": old_logp_batch
            }

        (loss, loss_a, loss_c, loss_e, kl, v_pred), grads = self.grad_fn(obs_batch, act_batch, ret_batch, adv_batch,
                                                                         old_dists_param)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "actor-loss": loss_a.asnumpy(),
            "critic-loss": loss_c.asnumpy(),
            "entropy": loss_e.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "kl": kl.asnumpy(),
            "predict_value": v_pred.mean().asnumpy()
        }

        return info
