"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
import numpy as np
from xuance.torch.learners import Learner
from argparse import Namespace


class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(SAC_Learner, self).__init__(config, policy, callback)
        self.optimizer = {
            'actor': torch.optim.Adam(self.policy.actor_parameters, self.config.learning_rate_actor),
            'critic': torch.optim.Adam(self.policy.critic_parameters, self.config.learning_rate_critic)}
        self.scheduler = {
            'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
            'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)}
        self.mse_loss = nn.MSELoss()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(policy.action_space.shape).item()
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate_actor)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        # actor update
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
        policy_q = torch.min(policy_q_1, policy_q_2).reshape([-1])
        p_loss = (self.alpha * log_pi.reshape([-1]) - policy_q).mean()
        self.optimizer['actor'].zero_grad()
        p_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
        self.optimizer['actor'].step()

        # critic update
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_value = target_q - self.alpha * log_pi_next.reshape([-1])
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        q_loss = self.mse_loss(action_q_1, backup.detach()) + self.mse_loss(action_q_2, backup.detach())
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros([])

        if self.scheduler is not None:
            self.scheduler['actor'].step()
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"Qloss/rank_{self.rank}": q_loss.item(),
                f"Ploss/rank_{self.rank}": p_loss.item(),
                f"Qvalue/rank_{self.rank}": policy_q.mean().item(),
                f"actor_lr/rank_{self.rank}": actor_lr,
                f"critic_lr/rank_{self.rank}": critic_lr,
            })
        else:
            info.update({
                "Qloss": q_loss.item(),
                "Ploss": p_loss.item(),
                "Qvalue": policy_q.mean().item(),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
            })
        if self.use_automatic_entropy_tuning:
            if self.distributed_training:
                info.update({f"alpha_loss/rank_{self.rank}": alpha_loss.item(),
                             f"alpha/rank_{self.rank}": self.alpha.item()})
            else:
                info.update({"alpha_loss": alpha_loss.item(),
                             "alpha": self.alpha.item()})

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                log_pi=log_pi, policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                policy_q=policy_q, p_loss=p_loss,
                                                action_q_1=action_q_1, action_q_2=action_q_2,
                                                log_pi_next=log_pi_next, target_q=target_q,
                                                target_value=target_value, backup=backup, q_loss=q_loss,
                                                alpha_loss=alpha_loss, alpha=self.alpha))
        return info
