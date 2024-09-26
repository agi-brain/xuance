"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace
from xuance.torch.utils.operations import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(PPG_Learner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0,
                                                           total_iters=self.config.running_steps)
        self.mse_loss = nn.MSELoss()
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0

    def update_policy(self, **samples):
        self.policy_iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_logp_batch = old_dist.log_prob(act_batch).detach()

        outputs, a_dist, _, _ = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss
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

        if self.distributed_training:
            info = {
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"clip_ratio/rank_{self.rank}": cr,
            }
        else:
            info = {
                "actor-loss": a_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "clip_ratio": cr,
            }

        return info

    def update_critic(self, **samples):
        self.value_iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)

        _, _, v_pred, _ = self.policy(obs_batch)
        loss = self.mse_loss(v_pred, ret_batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.distributed_training:
            info = {f"critic-loss/rank_{self.rank}": loss.item()}
        else:
            info = {"critic-loss": loss.item()}
        return info

    def update_auxiliary(self, **samples):
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])

        outputs, a_dist, v, aux_v = self.policy(obs_batch)
        aux_loss = self.mse_loss(v.detach(), aux_v)
        kl_loss = a_dist.kl_divergence(old_dist).mean()
        value_loss = self.mse_loss(v, ret_batch)
        loss = aux_loss + self.kl_beta * kl_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.distributed_training:
            info = {f"kl-loss/rank_{self.rank}": loss.item()}
        else:
            info = {"kl-loss": loss.item()}
        return info

    def update(self, *args):
        return
