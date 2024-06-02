"""
DQN with Quantile Regression (QRDQN)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11791
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional, Union
from argparse import Namespace


class QRDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        super(QRDQN_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = torch.as_tensor(samples['actions'], device=self.device).long()
        next_batch = samples['obs_next']
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], device=self.device)

        _, _, evalZ = self.policy(obs_batch)
        _, targetA, targetZ = self.policy(next_batch)

        current_quantile = (evalZ * self.one_hot(act_batch, evalZ.shape[1]).unsqueeze(-1)).sum(1)
        target_quantile = (targetZ * self.one_hot(targetA.detach(), evalZ.shape[1]).unsqueeze(-1)).sum(1).detach()
        target_quantile = rew_batch.unsqueeze(1) + self.gamma * target_quantile * (1 - ter_batch.unsqueeze(1))

        loss = self.mse_loss(target_quantile, current_quantile)
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

        info = {
            "Qloss": loss.item(),
            "learning_rate": lr,
        }

        return info
