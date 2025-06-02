"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(PG_Learner, self).__init__(config, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.ent_coef = config.ent_coef

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        ret_batch = torch.as_tensor(samples['returns'], device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch, returns=ret_batch)

        outputs, a_dist, _ = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        a_loss = -(ret_batch * log_prob).mean()
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

        if self.distributed_training:
            info.update({
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr
            })
        else:
            info.update({
                "actor-loss": a_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info, rep_output=outputs,
                                                a_dist=a_dist, log_prob=log_prob,
                                                a_loss=a_loss, e_loss=e_loss, loss=loss))
        return info
