"""
Advantage Actor-Critic (A2C)
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.torch.learners import Learner


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(A2C_Learner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0,
                                                           total_iters=config.running_steps)
        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
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

        if self.distributed_training:
            info = {
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"critic-loss/rank_{self.rank}": c_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"predict_value/rank_{self.rank}": v_pred.mean().item()
            }
        else:
            info = {
                "actor-loss": a_loss.item(),
                "critic-loss": c_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "predict_value": v_pred.mean().item()
            }

        return info
