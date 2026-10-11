"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: MindSpore
"""
import numpy as np
from argparse import Namespace

from xuance.mindspore import ms, ops, nn, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(SAC_Learner, self).__init__(config, model, callback)
        self.optimizer = {
            'actor': optim.Adam(params=self.model.actor.trainable_params(), lr=self.config.learning_rate, eps=1e-5),
            'critic': optim.Adam(params=self.model.critic.trainable_params(), lr=self.config.learning_rate, eps=1e-5),
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'], start_factor=1.0,
                                                 end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'], start_factor=1.0,
                                                  end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }
        self.mse_loss = nn.MSELoss()
        self._ones = ops.Ones()
        self.tau = config.tau
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(model.actor.action_space.shape).item()
            self.log_alpha = ms.Parameter(-self._ones(1, ms.float32))
            self.alpha = ops.exp(self.log_alpha)
            self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=config.learning_rate_actor)
            self.grad_fn_alpha = ms.value_and_grad(self.forward_fn_alpha, None, self.alpha_optimizer.parameters,
                                                   has_aux=True)
            self.grad_reducer_alpha = self.get_grad_reducer(self.alpha_optimizer)

        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)

        self.model.set_train()
        self.grad_reducer_actor = self.get_grad_reducer(self.optimizer["actor"])
        self.grad_reducer_critic = self.get_grad_reducer(self.optimizer["critic"])

    @ms.jit
    def training_step_alpha(self, log_pi):
        (alpha_loss, log_alpha), grads = self.grad_fn_alpha(log_pi)
        if self.distributed_training:
            grads = self.grad_reducer_alpha(grads)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.alpha_optimizer(grads)
        return alpha_loss, log_alpha

    @ms.jit
    def training_step_actor(self, obs_batch):
        (loss_a, log_pi, policy_q_1, policy_q_2, policy_q), grads = self.grad_fn_actor(obs_batch)
        if self.distributed_training:
            grads = self.grad_reducer_actor(grads)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer["actor"](grads)
        return loss_a, log_pi, policy_q_1, policy_q_2, policy_q

    @ms.jit
    def training_step_critic(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        (loss_q, action_q_1, action_q_2, log_pi_next, target_q, target_value, backup), grads = self.grad_fn_critic(
            obs_batch, act_batch, rew_batch, next_batch, ter_batch)
        if self.distributed_training:
            grads = self.grad_reducer_critic(grads)
        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer["critic"](grads)
        return loss_q, action_q_1, action_q_2, log_pi_next, target_q, target_value, backup

    def forward_fn_alpha(self, log_pi):
        alpha_loss = -ops.mean(self.log_alpha * (log_pi + self.target_entropy))
        return alpha_loss, self.log_alpha

    def forward_fn_actor(self, obs_batch):
        log_pi, policy_q_1, policy_q_2 = self.model.Qpolicy(obs_batch)
        policy_q = ops.minimum(policy_q_1, policy_q_2).reshape([-1])
        loss_a = ops.mean(self.alpha * log_pi.reshape([-1]) - policy_q)
        return loss_a, log_pi, policy_q_1, policy_q_2, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        action_q_1, action_q_2 = self.model.Qaction(obs_batch, act_batch)
        log_pi_next, target_q = self.model.Qtarget(next_batch)
        target_q = target_q.reshape([-1])
        target_value = target_q - self.alpha * log_pi_next
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        loss_q_1 = self.mse_loss(logits=action_q_1.reshape([-1]), labels=ops.stop_gradient(backup))
        loss_q_2 = self.mse_loss(logits=action_q_2.reshape([-1]), labels=ops.stop_gradient(backup))
        loss_q = loss_q_1 + loss_q_2
        return loss_q, action_q_1, action_q_2, log_pi_next, target_q, target_value, backup

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'], dtype=ms.float32)
        act_batch = Tensor(samples['actions'], dtype=ms.float32)
        rew_batch = Tensor(samples['rewards'], dtype=ms.float32)
        next_batch = Tensor(samples['obs_next'], dtype=ms.float32)
        ter_batch = Tensor(samples['terminals'], dtype=ms.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        q_loss, action_q_1, action_q_2, log_pi_next, target_q, target_value, backup = self.training_step_critic(
            obs_batch, act_batch, rew_batch, next_batch, ter_batch)

        p_loss, log_pi, policy_q_1, policy_q_2, policy_q = self.training_step_actor(obs_batch)

        if self.use_automatic_entropy_tuning:
            alpha_loss, _ = self.training_step_alpha(log_pi)
            self.alpha = ops.exp(self.log_alpha)
        else:
            alpha_loss = 0

        self.model.soft_update(self.tau)

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        actor_lr = self.scheduler['actor'].get_last_lr()[0]
        critic_lr = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "Qloss": q_loss.asnumpy(),
            "Ploss": p_loss.asnumpy(),
            "Qvalue": policy_q.mean().asnumpy(),
            "actor_lr": actor_lr.asnumpy(),
            "critic_lr": critic_lr.asnumpy(),
            "alpha_loss": alpha_loss.asnumpy(),
            "alpha": self.alpha.asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                log_pi=log_pi, policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                policy_q=policy_q, p_loss=p_loss,
                                                action_q_1=action_q_1, action_q_2=action_q_2,
                                                log_pi_next=log_pi_next, target_q=target_q,
                                                target_value=target_value, backup=backup, q_loss=q_loss,
                                                alpha_loss=alpha_loss, alpha=self.alpha))

        return info
