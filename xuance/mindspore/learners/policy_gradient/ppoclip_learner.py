"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore.nn import MSELoss
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPOCLIP_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        # Parameters
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        # APIs
        self._clip_range = [Tensor(1.0 - self.clip_range), Tensor(1.0 + self.clip_range)]
        self.loss = MSELoss()
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, obs_batch, act_batch, old_logp_batch, adv_batch, ret_batch):
        outputs, act_dist, v_pred = self.policy(obs_batch)
        log_prob = act_dist.log_prob(act_batch)
        ratio = ops.exp(log_prob - old_logp_batch)
        surrogate1 = ms.ops.clip_by_value(ratio, self._clip_range[0], self._clip_range[1]) * adv_batch
        surrogate2 = adv_batch * ratio
        loss_a = -ops.minimum(surrogate1, surrogate2).mean()
        loss_c = self.loss(v_pred, ret_batch)
        loss_e = act_dist.entropy().mean()
        loss = loss_a - self.ent_coef * loss_e + self.vf_coef * loss_c
        return loss, loss_a, loss_c, loss_e, v_pred, ratio

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])
        old_logp_batch = Tensor(samples['aux_batch']['old_logp'])

        (loss, a_loss, c_loss, e_loss, v_pred, ratio), grads = self.grad_fn(obs_batch, act_batch, old_logp_batch,
                                                                            adv_batch, ret_batch)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        # Logger
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]

        info = {
            "actor_loss": a_loss.asnumpy(),
            "critic_loss": c_loss.asnumpy(),
            "entropy": e_loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "predict_value": v_pred.mean().asnumpy(),
            "clip_ratio": cr.asnumpy(),
        }

        return info
