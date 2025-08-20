"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: MindSpore
"""
import numpy as np
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(SAC_Learner, self).__init__(config, policy)
        self.optimizer = {
            'actor': optim.Adam(params=self.policy.actor_parameters, lr=self.config.learning_rate, eps=1e-5),
            'critic': optim.Adam(params=self.policy.critic_parameters, lr=self.config.learning_rate, eps=1e-5),
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'], start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'], start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }
        self.mse_loss = nn.MSELoss()
        self._ones = ops.Ones()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(policy.action_space.shape).item()
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
        alpha_loss = -ops.mean(self.log_alpha * (log_pi + self.target_entropy))
        return alpha_loss, self.log_alpha

    def forward_fn_actor(self, x):
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(x)
        policy_q = ops.minimum(policy_q_1, policy_q_2).reshape([-1])
        loss_a = ops.mean(self.alpha * log_pi.reshape([-1]) - policy_q)
        return loss_a, log_pi, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_q = target_q.reshape([-1])
        target_value = target_q - self.alpha * log_pi_next
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        loss_q_1 = self.mse_loss(logits=action_q_1.reshape([-1]), labels=ops.stop_gradient(backup))
        loss_q_2 = self.mse_loss(logits=action_q_2.reshape([-1]), labels=ops.stop_gradient(backup))
        loss_q = loss_q_1 + loss_q_2
        return loss_q, action_q_1, action_q_2

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'], dtype=ms.float32)
        act_batch = Tensor(samples['actions'], dtype=ms.float32)
        rew_batch = Tensor(samples['rewards'], dtype=ms.float32)
        next_batch = Tensor(samples['obs_next'], dtype=ms.float32)
        ter_batch = Tensor(samples['terminals'], dtype=ms.float32)

        (q_loss, _, _), grads_critic = self.grad_fn_critic(obs_batch, act_batch, rew_batch, next_batch, ter_batch)
        if self.use_grad_clip:
            grads_critic = ops.clip_by_norm(grads_critic, self.grad_clip_norm)
        self.optimizer['critic'](grads_critic)

        (p_loss, log_pi, policy_q), grads_actor = self.grad_fn_actor(obs_batch)
        if self.use_grad_clip:
            grads_actor = ops.clip_by_norm(grads_actor, self.grad_clip_norm)
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
            "alpha_loss": alpha_loss.asnumpy(),
            "alpha": self.alpha.asnumpy(),
        }

        return info
