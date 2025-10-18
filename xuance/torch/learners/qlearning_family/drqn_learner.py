"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 callback):
        super(DRQN_Learner, self).__init__(config, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], dtype=torch.int64, device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        batch_size = samples['batch_size']
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             rew=rew_batch, termination=ter_batch, batch_size=batch_size)

        rnn_hidden = self.policy.init_hidden(batch_size)
        _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
        target_rnn_hidden = self.policy.init_hidden(batch_size)
        _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
        # targetQ = targetQ.max(dim=-1).values

        predictQ = evalQ.gather(-1, act_batch.unsqueeze(-1)).squeeze(-1)
        targetQ = targetQ.gather(-1, targetA.unsqueeze(-1)).squeeze(-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        loss = self.mse_loss(predictQ, targetQ.detach())
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
                f"learning_rate/rank_{self.rank}": lr,
                f"predictQ/rank_{self.rank}": predictQ.mean().item()
            })
        else:
            info.update({
                "Qloss": loss.item(),
                "learning_rate": lr,
                "predictQ": predictQ.mean().item()
            })
        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                evalQ=evalQ, predictQ=predictQ, targetA=targetA, targetQ=targetQ,
                                                loss=loss))
        return info
