"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, msd, ops, Module, Tensor, optim
from xuance.mindspore.utils import merge_distributions
from xuance.mindspore.learners import Learner


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPG_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
                                                     end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()

        # Get gradient function
        self.grad_fn_policy = ms.value_and_grad(self.forward_fn_policy, None, self.optimizer.parameters, has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer.parameters, has_aux=True)
        self.grad_fn_auxiliary = ms.value_and_grad(self.forward_fn_auxiliary, None, self.optimizer.parameters,
                                                   has_aux=True)
        self.policy.set_train()

    def forward_fn_policy(self, obs_batch, act_batch, adv_batch, old_log_prob_batch):
        if self.is_continuous:
            _, mu, std, _, _ = self.policy(obs_batch)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = log_prob.squeeze(-1)
            entropy = self.a_dist._entropy(mean=mu, sd=std)
            entropy = entropy.squeeze(-1)
        else:
            _, logits, _, _ = self.policy(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)
            entropy = self.a_dist.entropy(probs=probs)
        # ppo-clip core implementations
        ratio = ops.exp(log_prob - old_log_prob_batch)
        surrogate1 = ops.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio

        a_loss = -ops.mean(ops.minimum(surrogate1, surrogate2))
        e_loss = ops.mean(entropy)

        loss = a_loss - self.ent_coef * e_loss
        return loss, a_loss, e_loss, ratio

    def forward_fn_critic(self, obs_batch, ret_batch):
        if self.is_continuous:
            _, _, _, v_pred, _ = self.policy(obs_batch)
        else:
            _, _, v_pred, _ = self.policy(obs_batch)

        loss = self.mse_loss(v_pred, ops.stop_gradient(ret_batch))
        return loss, v_pred

    def forward_fn_auxiliary(self, *args):
        if self.is_continuous:
            obs_batch, ret_batch, old_mu, old_std = args
            _, mu, std, v, aux_v = self.policy(obs_batch)
            # calculate kl divergence
            kl = self.a_dist._kl_loss("Normal", mean_b=old_mu, sd_b=old_std, mean=mu, sd=std)
        else:
            obs_batch, ret_batch, old_logits = args
            _, logits, v, aux_v = self.policy(obs_batch)
            # calculate kl divergence
            old_probs = self.softmax(old_logits)
            probs = self.softmax(logits)
            kl = self.a_dist._kl_loss("Categorical", probs_b=old_probs, probs=probs)

        aux_loss = self.mse_loss(ops.stop_gradient(v), aux_v)
        kl_loss = ops.reduce_mean(kl)
        value_loss = self.mse_loss(v, ret_batch)

        loss = aux_loss + self.kl_beta * kl_loss + value_loss
        return loss, aux_loss, kl_loss, value_loss

    def update_policy(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        adv_batch = Tensor(samples['advantages'])
        if self.is_continuous:
            act_batch = Tensor(samples['actions'], dtype=ms.float32)
        else:
            act_batch = Tensor(samples['actions'], dtype=ms.int32)
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        old_log_prob_batch = ops.stop_gradient(old_dists.log_prob(act_batch))

        (loss, a_loss, e_loss, ratio), grads = self.grad_fn_policy(obs_batch, act_batch, adv_batch, old_log_prob_batch)
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "actor-loss": a_loss.asnumpy(),
            "entropy": e_loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "clip_ratio": ratio.mean().asnumpy(),
        }
        self.policy_iterations += 1

        return info

    def update_critic(self, **samples):
        obs_batch = Tensor(samples['obs'])
        ret_batch = Tensor(samples['returns'])

        (loss, v_pred), grads = self.grad_fn_critic(obs_batch, ret_batch)
        self.optimizer(grads)

        info = {
            "critic-loss": loss.asnumpy()
        }
        self.value_iterations += 1
        return info

    def update_auxiliary(self, **samples):
        obs_batch = Tensor(samples['obs'], dtype=ms.float32)
        ret_batch = Tensor(samples['returns'], dtype=ms.float32)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        if self.is_continuous:
            old_mu = old_dist.mu
            old_std = old_dist.std
            (loss, aux_loss, kl_loss, value_loss), grads = self.grad_fn_auxiliary(obs_batch, ret_batch, old_mu, old_std)
            self.optimizer(grads)
        else:
            old_logits = old_dist.logits
            (loss, aux_loss, kl_loss, value_loss), grads = self.grad_fn_auxiliary(obs_batch, ret_batch, old_logits)
            self.optimizer(grads)

        info = {
            "aux-loss": aux_loss.asnumpy(),
            "kl-loss": kl_loss.asnumpy(),
            "value-loss": value_loss.asnumpy(),
            "loss": loss.asnumpy()
        }
        return info

    def update(self, *args):
        return
