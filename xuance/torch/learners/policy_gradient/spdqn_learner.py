"""
Split parameterised deep Q network (SP-DQN)
Paper link: https://arxiv.org/pdf/1810.06394.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from typing import Optional, Union
from argparse import Namespace


class SPDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        super(SPDQN_Learner, self).__init__(config, episode_length, policy, optimizer, scheduler)
        self.tau = config.tau
        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        hyact_batch = torch.as_tensor(samples['actions'], device=self.device)
        disact_batch = hyact_batch[:, 0].long()
        conact_batch = hyact_batch[:, 1:]
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], device=self.device)

        # optimize Q-network
        with torch.no_grad():
            target_conact = self.policy.Atarget(next_batch)
            target_q = self.policy.Qtarget(next_batch, target_conact)
            target_q = torch.max(target_q, 1, keepdim=True)[0].squeeze()

            target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

        eval_qs = self.policy.Qeval(obs_batch, conact_batch)
        eval_q = eval_qs.gather(1, disact_batch.view(-1, 1)).squeeze()
        q_loss = self.mse_loss(eval_q, target_q)

        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # optimize actor network
        policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = - policy_q.mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        info = {
            "Q_loss": q_loss.item(),
            "P_loss": q_loss.item(),
            'Qvalue': eval_q.mean().item()
        }

        return info
