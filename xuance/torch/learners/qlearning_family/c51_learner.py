"""
Distributional Reinforcement Learning (C51DQN)
Paper link: http://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class C51_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(C51_Learner, self).__init__(config, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.atom_num = self.policy.atom_num

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'].reshape(-1, 1, 1), dtype=torch.int64, device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        _, _, evalZ = self.policy(obs_batch)
        _, targetA, targetZ = self.policy.target(next_batch)

        current_dist = evalZ.gather(1, act_batch.expand([-1, -1, self.atom_num])).squeeze(1)
        target_dist = targetZ.gather(1, targetA.reshape([-1, 1, 1]).expand([-1, -1, self.atom_num])).squeeze(1).detach()

        current_supports = self.policy.supports
        next_supports = rew_batch.unsqueeze(1) + self.gamma * self.policy.supports * (1 - ter_batch.unsqueeze(1))
        next_supports = next_supports.clamp(self.policy.v_min, self.policy.v_max)

        projection = 1 - (next_supports.unsqueeze(-1) - current_supports.unsqueeze(0)).abs() / self.policy.deltaz
        target_dist = torch.bmm(target_dist.unsqueeze(1), projection.clamp(0, 1)).squeeze(1)
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"Qloss/rank_{self.rank}": loss.item(),
                f"learning_rate/rank_{self.rank}": lr
            })
        else:
            info.update({
                "Qloss": loss.item(),
                "learning_rate": lr
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                evalZ=evalZ, targetA=targetA, targetZ=targetZ,
                                                current_dist=current_dist, target_dist=target_dist,
                                                current_supports=current_supports, next_supports=next_supports,
                                                projection=projection, loss=loss))
        return info
