"""
Soft Actor-Critic with discrete action spaces (SAC-Discrete)
Paper link: https://arxiv.org/pdf/1910.07207.pdf
Implementation: MindSpore
"""
from xuance.common import Optional
from xuance.mindspore import ms, Module, Tensor, optim, ops
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads
from argparse import Namespace
from mindspore.nn import MSELoss


class SACDIS_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 target_entropy: Optional[float] = None):
        super(SACDIS_Learner, self).__init__(config, policy)
        self.optimizer = {
            'actor': optim.Adam(params=self.policy.actor_parameters, lr=self.config.learning_rate, eps=1e-5),
            'critic': optim.Adam(params=self.policy.critic_parameters, lr=self.config.learning_rate, eps=1e-5),
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'], start_factor=1.0, end_factor=0.5,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'], start_factor=1.0, end_factor=0.5,
                                                  total_iters=self.config.running_steps)
        }
        self.mse_loss = MSELoss()
        self._ones = ops.Ones()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = target_entropy
            self.log_alpha = ms.Parameter(-self._ones(1, ms.float32))
            self.alpha = ops.exp(self.log_alpha)
            self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=config.learning_rate_actor)
            self.grad_fn_alpha = ms.value_and_grad(self.forward_fn_alpha, None, self.alpha_optimizer.parameters,
                                                   has_aux=True)
        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)

        self.policy.set_train()

    def forward_fn_alpha(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()
        return alpha_loss, self.log_alpha

    def forward_fn_actor(self, obs_batch):
        action_prob, log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
        policy_q = ops.minimum(policy_q_1, policy_q_2)
        p_loss = (action_prob * (self.alpha * log_pi - policy_q)).sum(axis=1).mean()
        return p_loss, log_pi, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, backup):
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch)
        action_q_1 = ops.gather(action_q_1, act_batch.long(), axis=-1, batch_dims=-1)
        action_q_2 = ops.gather(action_q_2, act_batch.long(), axis=-1, batch_dims=-1)
        q_loss = self.mse_loss(action_q_1, backup) + self.mse_loss(action_q_2, backup)
        return q_loss, action_q_1, action_q_2

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals']).reshape(-1, 1)
        act_batch = ops.expand_dims(act_batch, -1)

        action_prob_next, log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_q = action_prob_next * (target_q - self.alpha * log_pi_next)
        target_q = ops.expand_dims(target_q.sum(axis=1), -1)
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_q

        (q_loss, _, _), grads_critic = self.grad_fn_critic(obs_batch, act_batch, backup)
        if self.use_grad_clip:
            grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grads_critic)

        (p_loss, log_pi, policy_q), grads_actor = self.grad_fn_actor(obs_batch)
        if self.use_grad_clip:
            grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['actor'](grads_actor)

        if self.use_automatic_entropy_tuning:
            (alpha_loss, _), grads_alpha = self.grad_fn_alpha(log_pi)
            self.alpha_optimizer(grads_alpha)
            self.alpha = ops.exp(self.log_alpha)
        else:
            alpha_loss = 0

        self.policy.soft_update(self.tau)

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        actor_lr = self.scheduler['actor'].get_last_lr()[0]
        critic_lr = self.scheduler['critic'].get_last_lr()[0]

        info = {
            "Qloss": q_loss.asnumpy(),
            "Ploss": p_loss.asnumpy(),
            "Qvalue": policy_q.mean().asnumpy(),
            "actor_lr": actor_lr.asnumpy(),
            "critic_lr": critic_lr.asnumpy(),
        }
        if self.use_automatic_entropy_tuning:
            info.update({
                "alpha_loss": alpha_loss.asnumpy(),
                "alpha": self.alpha.asnumpy(),
            })

        return info
