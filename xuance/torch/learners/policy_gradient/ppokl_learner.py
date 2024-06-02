"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: Pytorch
"""
import torch
import numpy as np
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional, Union
from argparse import Namespace
from xuance.torch.utils.operations import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        super(PPOKL_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.target_kl = config.target_kl
        self.kl_coef = config.kl_coef

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        old_dists = samples['aux_batch']['old_dist']

        _, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        old_dist = merge_distributions(old_dists)
        kl = a_dist.kl_divergence(old_dist).mean()
        old_logp_batch = old_dist.log_prob(act_batch)

        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        a_loss = -(ratio * adv_batch).mean() + self.kl_coef * kl
        c_loss = self.mse_loss(v_pred, ret_batch)
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        if kl > self.target_kl * 1.5:
            self.kl_coef = self.kl_coef * 2.
        elif kl < self.target_kl * 0.5:
            self.kl_coef = self.kl_coef / 2.
        self.kl_coef = np.clip(self.kl_coef, 0.1, 20)
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
            "kl": kl.item(),
            "predict_value": v_pred.mean().item()
        }

        return info

