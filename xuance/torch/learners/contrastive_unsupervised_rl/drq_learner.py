"""
Deep Q-Network (DQN)
Paper link: https://www.nature.com/articles/nature14236
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace

from torchvision.transforms import transforms

from xuance.torch.learners import Learner

class FrameStackTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomPerspective(0.2),
            transforms.RandomRotation(15),
        ])

    def __call__(self, x: torch.Tensor):
        # x: (B, H, W, C), C=4
        x = x.permute(0,3,1,2)
        B, C, H, W = x.shape
        x_merged = x.contiguous().reshape(B*C, 1, H, W)  # (B*C,1,H,W)
        x_transformed = self.transform(x_merged)
        _, _, new_H, new_W = x_transformed.shape
        x_transformed = x_transformed.view(B, C, new_H, new_W)
        return x_transformed.permute(0,2,3,1)


class DrQ_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(DrQ_Learner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim
        self.transform = FrameStackTransform()
        self.aug_factor = config.aug_factor

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)

        aug_evalQ = []
        aug_targetQ = []
        for _ in range(self.aug_factor):
            obs_batch = self.transform(obs_batch)
            next_batch = self.transform(next_batch)
            _, _, evalQ = self.policy(obs_batch)
            _, _, targetQ = self.policy.target(next_batch)
            aug_evalQ.append(evalQ)
            aug_targetQ.append(targetQ)

        evalQ = torch.stack(aug_evalQ, dim=0)
        targetQ = torch.stack(aug_targetQ, dim=0)

        evalQ = evalQ.mean(dim=0)

        targetQ = targetQ.mean(dim=0)
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

        if self.distributed_training:
            info = {
                f"Qloss/rank_{self.rank}": loss.item(),
                f"predictQ/rank_{self.rank}": predictQ.mean().item(),
                f"learning_rate/rank_{self.rank}": lr,
            }
        else:
            info = {
                "Qloss": loss.item(),
                "predictQ": predictQ.mean().item(),
                "learning_rate": lr,
            }
        print(info)
        return info
