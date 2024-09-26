"""
Multi-pass parameterised deep Q network (MP-DQN)
Paper link: https://arxiv.org/pdf/1905.04388.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class MPDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(MPDQN_Learner, self).__init__(config, policy)
        conactor_optimizer = torch.optim.Adam(self.policy.conactor.parameters(), self.config.learning_rate)
        qnetwork_optimizer = torch.optim.Adam(self.policy.qnetwork.parameters(), self.config.learning_rate)
        self.optimizers = [conactor_optimizer, qnetwork_optimizer]
        conactor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(conactor_optimizer, start_factor=1.0, end_factor=0.25,
                                                                  total_iters=self.config.running_steps)
        qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(qnetwork_optimizer, start_factor=1.0, end_factor=0.25,
                                                                  total_iters=self.config.running_steps)
        self.scheduler = [conactor_lr_scheduler, qnetwork_lr_scheduler]
        self.tau = config.tau
        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        hyact_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        disact_batch = hyact_batch[:, 0].long()
        conact_batch = hyact_batch[:, 1:]

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

        if self.distributed_training:
            info = {
                f"Q_loss/rank_{self.rank}": q_loss.item(),
                f"P_loss/rank_{self.rank}": q_loss.item(),
                f"Qvalue/rank_{self.rank}": eval_q.mean().item()
            }
        else:
            info = {
                "Q_loss": q_loss.item(),
                "P_loss": q_loss.item(),
                "Qvalue": eval_q.mean().item()
            }

        return info
