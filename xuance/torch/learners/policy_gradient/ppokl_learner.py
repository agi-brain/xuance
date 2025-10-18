"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: Pytorch
"""
import torch
import numpy as np
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace
from xuance.torch.utils import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(PPOKL_Learner, self).__init__(config, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.target_kl = config.target_kl
        self.kl_coef = config.kl_coef

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        buffer_size = self.config.horizon_size * self.config.parallels
        update_times = self.config.running_steps // buffer_size
        total_iters = update_times * self.config.n_epochs * self.config.n_minibatch
        return total_iters

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        adv_batch = torch.as_tensor(samples['advantages'], device=self.device)
        old_dists = samples['aux_batch']['old_dist']
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch, old_dists=old_dists)

        outputs, a_dist, v_pred = self.policy(obs_batch)
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

        if self.distributed_training:
            info.update({
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"critic-loss/rank_{self.rank}": c_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"kl/rank_{self.rank}": kl.item(),
                f"predict_value/rank_{self.rank}": v_pred.mean().item()
            })
        else:
            info.update({
                "actor-loss": a_loss.item(),
                "critic-loss": c_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "kl": kl.item(),
                "predict_value": v_pred.mean().item()
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info, rep_output=outputs,
                                                a_dist=a_dist, v_pred=v_pred, log_prob=log_prob, kl=kl, ratio=ratio,
                                                a_loss=a_loss, c_loss=c_loss, e_loss=e_loss, loss=loss))
        return info

