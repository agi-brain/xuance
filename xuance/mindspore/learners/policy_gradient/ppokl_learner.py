"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
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

    def forward_fn(self, obs_batch, act_batch, adv_batch, old_logp_batch):
        outputs, act_dist, v_pred = self.policy(obs_batch)
        log_prob = act_dist.log_prob(act_batch)
        ratio = self._exp(log_prob - old_log_p)
        surrogate1 = ms.ops.clip_by_value(ratio, self._clip_range[0], self._clip_range[1]) * adv
        surrogate2 = adv * ratio
        loss_a = -self._mean(self._minimum(surrogate1, surrogate2))
        loss_c = self._loss(logits=v_pred, labels=ret)
        loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
        loss = loss_a - self._ent_coef * loss_e + self._vf_coef * loss_c
        return loss

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
