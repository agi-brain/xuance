"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace
from xuance.mindspore.utils.operations import merge_distributions
from mindspore.nn import MSELoss


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPG_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.mse_loss = MSELoss()
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0
        # Get gradient function
        self._mean = ms.ops.ReduceMean(keep_dims=True)
        self._minimum = ms.ops.Minimum()
        self._exp = ms.ops.Exp()
        self.grad_fn_policy = ms.value_and_grad(self.forward_fn_policy, None, self.optimizer.parameters, has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer.parameters, has_aux=True)
        self.grad_fn_auxiliary = ms.value_and_grad(self.forward_fn_auxiliary, None, self.optimizer.parameters,
                                                   has_aux=True)
        self.policy.set_train()
        self.continuous_control = True if ("gaussian" in str(type(self.policy))) else False

    def forward_fn_policy(self, obs_batch, act_batch, adv_batch, old_logp_batch):
        _, a_dist, _, _ = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        # ppo-clip core implementations
        ratio = self._exp(log_prob - old_logp_batch)
        surrogate1 = ms.ops.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -self._minimum(surrogate1, surrogate2).mean()
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss
        return loss, a_loss, e_loss, ratio

    def forward_fn_critic(self, obs_batch, ret_batch):
        _, _, v_pred, _ = self.policy(obs_batch)
        loss = self.mse_loss(v_pred, ret_batch)
        return loss, v_pred

    def forward_fn_auxiliary(self, obs_batch, ret_batch, old_dists_param):
        _, a_dist, v, aux_v = self.policy(obs_batch)
        aux_loss = self.mse_loss(v, aux_v)
        if self.continuous_control:
            mean, std = old_dists_param['mean'], old_dists_param['std']
            kl_loss = a_dist.distribution.kl_loss('Normal', mean, std).mean()
        else:
            probs = old_dists_param['probs']
            kl_loss = a_dist.distribution.kl_loss('Categorical', probs).mean()
        value_loss = self.mse_loss(v, ret_batch)
        loss = aux_loss + self.kl_beta * kl_loss + value_loss
        return loss, v

    def update_policy(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        adv_batch = Tensor(samples['advantages'])
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_logp_batch = old_dist.log_prob(act_batch)

        (loss, a_loss, e_loss, ratio), grads = self.grad_fn_policy(obs_batch, act_batch, adv_batch, old_logp_batch)
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
        obs_batch = samples['obs']
        ret_batch = Tensor(samples['returns'])
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        if self.continuous_control:
            old_dists_param = {
                "mean": old_dist.mu,
                "std": old_dist.std
            }
        else:
            old_dists_param = {
                "probs": old_dist.probs
            }

        (loss, v), grads = self.grad_fn_auxiliary(obs_batch, ret_batch, old_dists_param)
        self.optimizer(grads)

        info = {
            "kl-loss": loss.asnumpy()
        }
        return info

    def update(self, *args):
        return

