"""
Multi-Agent TD3
"""
import torch
import numpy as np
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.torch import Tensor
from typing import Optional, List
from argparse import Namespace
from operator import itemgetter


class MATD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        self.actor_update_delay = config.actor_update_delay
        super(MATD3_Learner, self).__init__(config, model_keys, agent_keys, policy, optimizer, scheduler)
        self.optimizer = {key: {'actor': optimizer[key][0],
                                'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0],
                                'critic': scheduler[key][1]} for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        if self.use_parameter_sharing:
            k = self.model_keys[0]
            obs = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['obs'])[i][:, None]
                                             for i in range(self.n_agents)], axis=1)).to(self.device)}
            actions = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['actions'])[i][:, None]
                                                 for i in range(self.n_agents)], axis=1)).to(self.device)}
            obs_next = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['obs_next'])[i][:, None]
                                                  for i in range(self.n_agents)], axis=1)).to(self.device)}
            rewards = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['rewards'])[i][:, None, None]
                                                 for i in range(self.n_agents)], axis=1)).to(self.device)}
            terminals = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['terminals'])[i][:, None, None]
                                                   for i in range(self.n_agents)], axis=1)).to(self.device)}
            agent_mask = {k: Tensor(np.concatenate([itemgetter(*self.agent_keys)(sample['agent_mask'])[i][:, None, None]
                                                    for i in range(self.n_agents)], axis=1)).to(self.device)}
            IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.config.batch_size, -1, -1).to(self.device)
        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
            rewards = {k: Tensor(sample['rewards'][k][:, None]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k][:, None]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k][:, None]).float().to(self.device) for k in self.agent_keys}
            IDs = None

        # train the model
        if self.iterations % self.actor_update_delay == 0:  # update actor(s)
            actions_eval = self.policy(obs, IDs)
            for key in self.model_keys:
                # update actor
                actions_eval_detach_others = {}
                for k in self.model_keys:
                    if k == key:
                        actions_eval_detach_others[k] = actions_eval[key]
                    else:
                        actions_eval_detach_others[k] = actions_eval[key].detach()

                _, _, q_policy = self.policy.Qpolicy(obs, actions_eval_detach_others, IDs, key)
                loss_a = -(q_policy[key] * agent_mask[key]).sum() / agent_mask[key].sum()
                self.optimizer[key]['actor'].zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
                self.optimizer[key]['actor'].step()
                if self.scheduler[key]['actor'] is not None:
                    self.scheduler[key]['actor'].step()

                lr_a = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']

                info.update({
                    f"{key}/learning_rate_actor": lr_a,
                    f"{key}/loss_actor": loss_a.item(),
                })

        # update critic(s)
        actions_next = self.policy.Atarget(obs_next, IDs)
        for key in self.model_keys:
            q_eval_A, q_eval_B, _ = self.policy.Qpolicy(obs, actions, IDs, key)
            q_next = self.policy.Qtarget(obs_next, actions_next, IDs, key)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next[key]
            td_error_A = (q_eval_A[key] - q_target.detach()) * agent_mask[key]
            td_error_B = (q_eval_B[key] - q_target.detach()) * agent_mask[key]
            loss_c = ((td_error_A ** 2).sum() + (td_error_B ** 2).sum()) / agent_mask[key].sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            lr_c = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_critic": lr_c,
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ_A": q_eval_A[key].mean().item(),
                f"{key}/predictQ_B": q_eval_B[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info
