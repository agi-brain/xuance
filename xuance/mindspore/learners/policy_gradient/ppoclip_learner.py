"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, msd, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPOCLIP_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        # Parameters
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch, old_log_prob_batch):
        if self.is_continuous:
            outputs, mu, std, v_pred = self.policy(obs_batch)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = log_prob.squeeze(-1)
            entropy = self.a_dist._entropy(mean=mu, sd=std)
            entropy = entropy.squeeze(-1)
        else:
            outputs, logits, v_pred = self.policy(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)
            entropy = self.a_dist.entropy(probs=probs)
        ratio = ops.exp(log_prob - old_log_prob_batch)
        surrogate1 = ops.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -ops.mean(ops.minimum(surrogate1, surrogate2))
        c_loss = self.mse_loss(logits=v_pred, labels=ops.stop_gradient(ret_batch))
        e_loss = ops.mean(entropy)
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        return loss, a_loss, c_loss, e_loss, v_pred, ratio

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'], dtype=ms.float32)
        ret_batch = Tensor(samples['returns'], dtype=ms.float32)
        adv_batch = Tensor(samples['advantages'], dtype=ms.float32)
        old_log_prob_batch = Tensor(samples['aux_batch']['old_logp'], dtype=ms.float32)
        if self.is_continuous:
            act_batch = Tensor(samples['actions'], dtype=ms.float32)
        else:
            act_batch = Tensor(samples['actions'], dtype=ms.int32)

        (loss, a_loss, c_loss, e_loss, v_pred, ratio), grads = self.grad_fn(obs_batch, act_batch, ret_batch,
                                                                            adv_batch, old_log_prob_batch)
        if self.use_grad_clip:
            grads = ops.clip_by_norm(grads, self.grad_clip_norm)
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
