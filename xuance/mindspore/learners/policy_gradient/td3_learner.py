"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads
from argparse import Namespace
from mindspore.nn import MSELoss


class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(TD3_Learner, self).__init__(config, policy)
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
        self.tau = config.tau
        self.gamma = config.gamma
        self.actor_update_delay = config.actor_update_delay
        self.mse_loss = MSELoss()
        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)
        self.policy.set_train()

    def forward_fn_actor(self, obs_batch):
        policy_q = self.policy.Qpolicy(obs_batch)
        loss_p = -policy_q.mean()
        return loss_p, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, backup):
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        loss_q = self.mse_loss(logits=action_q_A, labels=backup) + self.mse_loss(logits=action_q_B, labels=backup)
        return loss_q, action_q_A, action_q_B

    def update(self, **samples):
        self.iterations += 1
        info = {}
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q

        (q_loss, action_q_A, action_q_B), grads_critic = self.grad_fn_critic(obs_batch, act_batch, target_q)
        if self.use_grad_clip:
            grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer['critic'](grads_critic)

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

        return info
