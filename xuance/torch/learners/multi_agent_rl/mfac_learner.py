"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import numpy as np
import torch
from torch import nn
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.torch import Tensor
from xuance.torch.utils import ValueNorm
from xuance.torch.learners.multi_agent_rl.iac_learner import IAC_Learner


class MFAC_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(MFAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)

    def update(self, sample):
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        act_mean = torch.Tensor(sample['act_mean']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        act_mean_n = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])

        info = self.callback.on_update_start(self.iterations, method="update", policy=self.policy)

        # actor loss
        _, pi_dist = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions).unsqueeze(-1)
        entropy = pi_dist.entropy().unsqueeze(-1)

        targets = returns
        value_pred = self.policy.critic(obs, act_mean_n, IDs)
        advantages = targets - value_pred
        td_error = value_pred - targets.detach()

        pg_loss = -((advantages.detach() * log_pi) * agent_mask).sum() / agent_mask.sum()
        vf_loss = ((td_error ** 2) * agent_mask).sum() / agent_mask.sum()
        entropy_loss = (entropy * agent_mask).sum() / agent_mask.sum()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "loss": loss.item(),
            "predicted_value": value_pred.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
