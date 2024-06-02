"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional
from argparse import Namespace


class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: dict,
                 scheduler: Optional[dict] = None,
                 target_entropy: Optional[float] = None):
        super(SAC_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.mse_loss = nn.MSELoss()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.actor_learning_rate)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = samples['obs_next']
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], device=self.device)

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
            alpha_loss = 0

        if self.scheduler is not None:
            self.scheduler['actor'].step()
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "Ploss": p_loss.item(),
            "Qvalue": policy_q.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

        return info
