"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional
from argparse import Namespace


class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: dict,
                 scheduler: Optional[dict] = None):
        super(TD3_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.tau = config.tau
        self.gamma = config.gamma
        self.actor_update_delay = config.actor_update_delay
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        info = {}
        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = samples['obs_next']
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], device=self.device)

        # critic update
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
        q_loss = self.mse_loss(action_q_A, target_q.detach()) + self.mse_loss(action_q_B, target_q.detach())
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()
        if self.scheduler is not None:
            self.scheduler['critic'].step()

        # actor update
        if self.iterations % self.actor_update_delay == 0:
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -policy_q.mean()
            self.optimizer['actor'].zero_grad()
            p_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
            self.optimizer['actor'].step()
            if self.scheduler is not None:
                self.scheduler['actor'].step()
            self.policy.soft_update(self.tau)
            info.update({"Ploss": p_loss.item()})

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info.update({
            "Qloss": q_loss.item(),
            "QvalueA": action_q_A.mean().item(),
            "QvalueB": action_q_B.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        })

        return info
