from argparse import Namespace
from operator import itemgetter
from typing import List, Optional

import numpy as np
import torch
from torch import nn, Tensor

from xuance.torch.agents import BaseCallback
from xuance.torch.learners import IQL_Learner


class DGN_Learner(IQL_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback: Optional[BaseCallback] = None,):
        super(DGN_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled = None, None, None
        obs_next, state_next, avail_actions_next = None, None, None
        IDs = None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.n_agents == 1:
                obs_tensor = Tensor(sample['obs'][k]).to(self.device).unsqueeze(1)
                actions_tensor = Tensor(sample['actions'][k]).to(self.device).unsqueeze(1)
                rewards_tensor = Tensor(sample['rewards'][k]).to(self.device).unsqueeze(1)
                ter_tensor = Tensor(sample['terminals'][k]).float().to(self.device).unsqueeze(1)
                msk_tensor = Tensor(sample['agent_mask'][k]).float().to(self.device).unsqueeze(1)
            else:
                obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']),
                                             axis=1)).to(self.device)
                actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']),
                                                 axis=1)).to(self.device)
                rewards_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['rewards']),
                                                 axis=1)).to(self.device)
                ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']),
                                             axis=1)).float().to(self.device)
                msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']),
                                             axis=1)).float().to(self.device)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape(batch_size, self.n_agents, seq_length + 1, -1)}
                if len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, seq_length)}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: actions_tensor.reshape(bs, seq_length, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents, seq_length)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents, seq_length)}
                agent_mask = {k: msk_tensor.reshape(bs, seq_length)}
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length + 1, -1).reshape(bs, seq_length + 1, self.n_agents).to(self.device)
            else:
                obs = {k: obs_tensor.reshape(batch_size, self.n_agents, -1)}
                if len(actions_tensor.shape) == 2:
                    actions = {k: actions_tensor.reshape(bs)}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                obs_next = {k: Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs_next']),
                                               axis=1)).to(self.device).reshape(bs, -1)}
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(
                    batch_size, -1, -1).reshape(bs, self.n_agents).to(self.device)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, seq_length + 1, -1])).float().to(self.device)}
                else:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, -1])).float().to(self.device)}
                    avail_a_next = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions_next']), axis=1)
                    avail_actions_next = {k: Tensor(avail_a_next.reshape([bs, -1])).float().to(self.device)}
        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            rewards = {k: Tensor(sample['rewards'][k]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
            if not self.use_rnn:
                obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}
                if not self.use_rnn:
                    avail_actions_next = {k: Tensor(sample['avail_actions_next'][k]).float().to(self.device) for k in self.model_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)
            if not self.use_rnn:
                state_next = Tensor(sample['state_next']).to(self.device)

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

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample_Tensor['seq_length']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            filled = filled.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs_rnn, seq_len)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents, seq_len)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents, seq_len)
        else:
            bs_rnn = batch_size

        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        alive_ally = {k: agent_mask[k].unsqueeze(dim=-1) for k in self.model_keys}
        zeros = torch.zeros(bs_rnn, 1, 1).to(self.config.device)
        alive_ally = {k: torch.cat([alive_ally[k], zeros], dim=1) for k in self.model_keys}
        if self.config.use_parameter_sharing:
            key = self.model_keys[0]
            alive_ally = {k: alive_ally[key][i * batch_size: i * batch_size + batch_size] for i, k in
                          enumerate(self.agent_keys)}
        _, actions_greedy, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                                rnn_hidden=rnn_hidden, alive_ally=alive_ally)
        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, q_next_seq = self.policy.Qtarget(observation=obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden,
                                            alive_ally=alive_ally)

        total_loss = 0
        for key in self.model_keys:
            q_eval_a = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next = q_next_seq[key][:, 1:]
            if self.use_actions_mask:
                q_next[avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                actions_next_greedy = actions_greedy[key][:, 1:].unsqueeze(-1)
                q_next_a = q_next.gather(-1, actions_next_greedy.long().detach()).reshape(bs_rnn, seq_len)
            else:
                q_next_a = q_next.max(dim=-1, keepdim=True).values.reshape(bs_rnn, seq_len)

            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

            # calculate the loss function
            mask_values = agent_mask[key] * filled
            td_errors = (q_eval_a - q_target.detach()) * mask_values
            loss = (td_errors ** 2).sum() / mask_values.sum()
            total_loss = total_loss + loss
            if self.scheduler is not None:
                self.scheduler[key].step()

            lr = self.optimizer[key].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate": lr,
                f"{key}/loss_Q": loss.item(),
                f"{key}/predictQ": q_eval_a.mean().item()
            })
        for opt in self.optimizer.values():
            opt.zero_grad()
        total_loss.backward()
        if self.use_grad_clip:
            for key in self.model_keys:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_model[key], self.grad_clip_norm)
        for opt in self.optimizer.values():
            opt.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info