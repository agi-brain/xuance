"""
Deep Q-Network (DQN)
Paper link: https://www.nature.com/articles/nature14236
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.torch.learners import Learner


class DQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(DQN_Learner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0,
                                                           total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        sample_Tensor = self.build_training_data(samples=samples,
                                                 use_distributed_training=self.distributed_training)
        obs_batch = sample_Tensor['obs']
        act_batch = sample_Tensor['actions']
        next_batch = sample_Tensor['obs_next']
        rew_batch = sample_Tensor['rewards']
        ter_batch = sample_Tensor['terminals']

        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(dim=-1).values
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * self.one_hot(act_batch.long(), evalQ.shape[1])).sum(dim=-1)

        loss = self.mse_loss(predictQ, targetQ)
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
            "predictQ": predictQ.mean().item(),
            "learning_rate": lr,
        }

        return info
