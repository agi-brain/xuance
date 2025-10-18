"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(TD3_Learner, self).__init__(config, policy, callback)
        self.optimizer = {
            'actor': optim.Adam(params=self.policy.actor_parameters, lr=self.config.learning_rate, eps=1e-5),
            'critic': optim.Adam(params=self.policy.critic_parameters, lr=self.config.learning_rate, eps=1e-5),
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'], start_factor=1.0,
                                                 end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'], start_factor=1.0,
                                                  end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }
        self.tau = config.tau
        self.gamma = config.gamma
        self.actor_update_delay = config.actor_update_delay
        self.mse_loss = nn.MSELoss()
        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)
        self.policy.set_train()

    def forward_fn_actor(self, obs_batch):
        policy_q = self.policy.Qpolicy(obs_batch).reshape([-1])
        loss_p = -ops.mean(policy_q)
        return loss_p, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q

        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        loss_q_A = self.mse_loss(logits=action_q_A, labels=ops.stop_gradient(target_q))
        loss_q_B = self.mse_loss(logits=action_q_B, labels=ops.stop_gradient(target_q))
        loss_q = loss_q_A + loss_q_B
        return loss_q, next_q, action_q_A, action_q_B, target_q

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        (q_loss, next_q, action_q_A, action_q_B, target_q), grads_critic = self.grad_fn_critic(
            obs_batch, act_batch, next_batch, rew_batch, ter_batch)
        if self.use_grad_clip:
            grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grads_critic)

        policy_q, p_loss = None, None
        if self.iterations % self.actor_update_delay == 0:
            (p_loss, policy_q), grads_actor = self.grad_fn_actor(obs_batch)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer['actor'](grads_actor)
            self.policy.soft_update(self.tau)
            info["Ploss"] = p_loss.asnumpy()

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        actor_lr = self.scheduler['actor'].get_last_lr()[0]
        critic_lr = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "Qloss": q_loss.asnumpy(),
            "QvalueA": action_q_A.mean().asnumpy(),
            "QvalueB": action_q_B.mean().asnumpy(),
            "actor_lr": actor_lr.numpy(),
            "critic_lr": critic_lr.numpy()
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                action_q_A=action_q_A, action_q_B=action_q_B,
                                                next_q=next_q, target_q=target_q, q_loss=q_loss,
                                                policy_q=policy_q, p_loss=p_loss))

        return info
