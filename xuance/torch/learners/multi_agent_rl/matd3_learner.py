"""
Multi-Agent TD3
"""
import numpy as np
import torch
from torch import nn
from xuance.torch.learners.multi_agent_rl.maddpg_learner import MADDPG_Learner
from typing import Optional, List
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor


class MATD3_Learner(MADDPG_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        super(MATD3_Learner, self).__init__(config, model_keys, agent_keys, episode_length,
                                            policy, optimizer, scheduler)
        self.actor_update_delay = config.actor_update_delay

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)).to(self.device)
        obs_critic = {key: obs_tensor.reshape(batch_size, -1) for key in self.model_keys}
        obs_next_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs_next']), axis=1)).to(self.device)
        obs_next_critic = {key: obs_next_tensor.reshape(batch_size, -1) for key in self.model_keys}

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:  # update actor(s)
            _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
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

    def update_rnn(self, *args):
        raise NotImplementedError
