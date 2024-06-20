"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: Pytorch
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
import numpy as np
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor


class MADDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(MADDPG_Learner, self).__init__(config, model_keys, agent_keys, episode_length,
                                             policy, optimizer, scheduler)
        self.optimizer = {key: {'actor': optimizer[key][0],
                                'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0],
                                'critic': scheduler[key][1]} for key in self.model_keys}
    
    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 0
        obs_next, filled, IDs = None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)).to(self.device)
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)).to(self.device)
            rewards_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['rewards']), axis=1)).to(self.device)
            ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1)).float().to(self.device)
            msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1)).float().to(self.device)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape(bs, seq_length + 1, -1)}
                actions = {k: actions_tensor.reshape(bs, seq_length, -1)}
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents, seq_length)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents, seq_length)}
                agent_mask = {k: msk_tensor.reshape(bs, seq_length)}
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length + 1, -1).reshape(bs, seq_length + 1, self.n_agents).to(self.device)
            else:
                obs = {k: obs_tensor.reshape(bs, -1)}
                actions = {k: actions_tensor.reshape(bs, -1)}
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                obs_next = {k: Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs_next']),
                                               axis=1)).to(self.device).reshape(bs, -1)}
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(
                    batch_size, -1, -1).reshape(bs, self.n_agents).to(self.device)
        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            rewards = {k: Tensor(sample['rewards'][k]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
            if not self.use_rnn:
                obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'state_next': state_next,
            'obs': obs,
            'actions': actions,
            'obs_next': obs_next,
            'rewards': rewards,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'avail_actions_next': avail_actions_next,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

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
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        # train the model
        _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
        _, actions_next = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        for key in self.model_keys:
            # update actor
            actions_detach_others = {}
            for k in self.model_keys:
                if k == key:
                    actions_detach_others[k] = actions_eval[key]
                else:
                    actions_detach_others[k] = actions_eval[key].detach()

            _, q_policy = self.policy.Qpolicy(observation=obs, actions=actions_detach_others, agent_ids=IDs,
                                              agent_key=key)
            q_policy_i = q_policy[key].reshape(bs)
            loss_a = -(q_policy_i * agent_mask[key]).sum() / agent_mask[key].sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            # update critic
            _, q_eval = self.policy.Qpolicy(observation=obs, actions=actions, agent_ids=IDs, agent_key=key)
            _, q_next = self.policy.Qtarget(next_observation=obs_next, next_actions=actions_next, agent_ids=IDs,
                                            agent_key=key)
            q_eval_a = q_eval[key].reshape(bs)
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
            td_error = (q_eval_a - q_target.detach()) * agent_mask[key]
            loss_c = (td_error ** 2).sum() / agent_mask[key].sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
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

    def update_rnn(self, *args):
        raise NotImplementedError
