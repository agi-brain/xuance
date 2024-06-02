"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional, Union
from argparse import Namespace


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        super(PPOCLIP_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range

    def update(self, **samples):
        self.iterations += 1

        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        old_logp_batch = torch.as_tensor(samples['aux_batch']['old_logp'], device=self.device)

        outputs, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()

        c_loss = self.mse_loss(v_pred, ret_batch.detach())

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
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item(),
            "clip_ratio": cr
        }

        return info
