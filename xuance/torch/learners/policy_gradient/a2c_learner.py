"""
Advantage Actor-Critic (A2C)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional, Union
from argparse import Namespace


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        self.mse_loss = nn.MSELoss()
        super(A2C_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef

    def update(self, **samples):
        self.iterations += 1

        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        outputs, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        a_loss = -(adv_batch * log_prob).mean()
        c_loss = self.mse_loss(v_pred, ret_batch)
        e_loss = a_dist.entropy().mean()

        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item()
        }

        return info
