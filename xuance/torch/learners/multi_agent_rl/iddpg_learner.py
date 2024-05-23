"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.torch import Tensor
from typing import Optional, List
from argparse import Namespace


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(IDDPG_Learner, self).__init__(config, model_keys, policy, optimizer, scheduler)
        self.use_parameter_sharing = config.use_parameter_sharing
        self.optimizer = {key: {'actor': optimizer[key][0],
                                'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0],
                                'critic': scheduler[key][1]} for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}
        if self.use_parameter_sharing:
            sample = {self.model_keys[0]: sample}
            IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.config.batch_size, -1, -1).to(self.device)
        else:
            IDs = None

        for key in self.model_keys:
            obs = {key: Tensor(sample[key]['obs']).to(self.device)}
            actions = {key: Tensor(sample[key]['actions']).to(self.device)}
            obs_next = {key: Tensor(sample[key]['obs_next']).to(self.device)}
            if self.use_parameter_sharing:
                rewards = Tensor(sample[key]['rewards']).reshape(-1, self.n_agents, 1).to(self.device)
                terminals = Tensor(sample[key]['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = Tensor(sample[key]['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
            else:
                rewards = Tensor(sample[key]['rewards']).reshape(-1, 1).to(self.device)
                terminals = Tensor(sample[key]['terminals']).float().reshape(-1, 1).to(self.device)
                agent_mask = Tensor(sample[key]['agent_mask']).float().reshape(-1, 1).to(self.device)

            q_eval = self.policy.Qpolicy(obs, actions, IDs, key)
            q_next = self.policy.Qtarget(obs_next, self.policy.Atarget(obs_next, IDs, key), IDs, key)
            q_target = rewards + (1 - terminals) * self.gamma * q_next[key]

            # calculate the loss function
            actions_eval = self.policy(obs, IDs, key)
            q_policy = self.policy.Qpolicy(obs, actions_eval, IDs, key)[key]
            loss_a = -(q_policy * agent_mask).sum() / agent_mask.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.config.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            td_error = (q_eval[key] - q_target.detach()) * agent_mask
            loss_c = (td_error ** 2).sum() / agent_mask.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.config.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            lr_a = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            lr_c = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": lr_a,
                f"{key}/learning_rate_critic": lr_c,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": q_eval[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info
