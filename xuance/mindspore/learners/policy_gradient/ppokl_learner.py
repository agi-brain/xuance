"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, msd, ops, Module, Tensor, optim
from xuance.mindspore.utils import merge_distributions
from xuance.mindspore.learners import Learner


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPOKL_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.target_kl = config.target_kl
        self.kl_coef = Tensor(config.kl_coef)
        self.clip_range = config.clip_range
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()

        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, *args):
        if self.is_continuous:
            obs_batch, act_batch, ret_batch, adv_batch, old_mu, old_std = args
            outputs, mu, std, v_pred = self.policy(obs_batch)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = log_prob.squeeze(-1)
            old_log_prob = self.a_dist._log_prob(value=act_batch, mean=old_mu, sd=old_std)
            old_log_prob = old_log_prob.squeeze(-1)
            entropy = self.a_dist._entropy(mean=mu, sd=std)
            entropy = entropy.squeeze(-1)
            kl = self.a_dist._kl_loss("Normal", mean_b=old_mu, sd_b=old_std, mean=mu, sd=std)
        else:
            obs_batch, act_batch, ret_batch, adv_batch, old_logits = args
            outputs, logits, v_pred = self.policy(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)
            old_probs = self.softmax(old_logits)
            old_log_prob = self.a_dist._log_prob(value=act_batch, probs=old_probs)
            entropy = self.a_dist.entropy(probs=probs)
            kl = self.a_dist._kl_loss("Categorical", probs_b=old_probs, probs=probs)

        ratio = ops.exp(log_prob - old_log_prob)
        kl = ops.reduce_mean(kl)
        a_loss = -ops.mean(ratio * adv_batch) + self.kl_coef * kl
        c_loss = self.mse_loss(logits=v_pred, labels=ops.stop_gradient(ret_batch))
        e_loss = ops.mean(entropy)
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        return loss, a_loss, c_loss, e_loss, kl, v_pred

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])
        if self.is_continuous:
            act_batch = Tensor(samples['actions'], dtype=ms.float32)
        else:
            act_batch = Tensor(samples['actions'], dtype=ms.int32)
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        if self.is_continuous:
            old_mu = old_dists.mu
            old_std = old_dists.std
            (loss, a_loss, c_loss, e_loss, kl, v_pred), grads = self.grad_fn(obs_batch, act_batch, ret_batch, adv_batch,
                                                                             old_mu, old_std)
        else:
            old_logits = old_dists.logits
            (loss, a_loss, c_loss, e_loss, kl, v_pred), grads = self.grad_fn(obs_batch, act_batch, ret_batch, adv_batch,
                                                                             old_logits)
        if self.use_grad_clip:
            grads = ops.clip_by_norm(grads, self.grad_clip_norm)
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        if kl.asnumpy() > self.target_kl * 1.5:
            self.kl_coef = self.kl_coef * 2.
        elif kl.asnumpy() < self.target_kl * 0.5:
            self.kl_coef = self.kl_coef / 2.
        self.kl_coef = ops.clip_by_value(self.kl_coef, 0.1, 20)

        info = {
            "actor-loss": a_loss.asnumpy(),
            "critic-loss": c_loss.asnumpy(),
            "entropy": e_loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "kl": kl.asnumpy(),
            "predict_value": v_pred.mean().asnumpy()
        }

        return info
