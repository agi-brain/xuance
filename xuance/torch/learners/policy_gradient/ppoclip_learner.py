"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(PPOCLIP_Learner, self).__init__(config, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range

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
        old_logp_batch = torch.as_tensor(samples['aux_batch']['old_logp'], device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch, old_logp=old_logp_batch)

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
        
        if self.distributed_training:
            info.update({
                f"actor_loss/rank_{self.rank}": a_loss.item(),
                f"critic_loss/rank_{self.rank}": c_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"predict_value/rank_{self.rank}": v_pred.mean().item(),
                f"clip_ratio/rank_{self.rank}": cr
            })
        else:
            info.update({
                "actor_loss": a_loss.item(),
                "critic_loss": c_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "predict_value": v_pred.mean().item(),
                "clip_ratio": cr
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info, rep_output=outputs,
                                                a_dist=a_dist, v_pred=v_pred, log_prob=log_prob,
                                                ratio=ratio, surrogate1=surrogate1, surrogate2=surrogate2,
                                                a_loss=a_loss, c_loss=c_loss, e_loss=e_loss, loss=loss))
        return info
