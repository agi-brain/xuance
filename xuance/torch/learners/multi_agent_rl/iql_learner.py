"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = {key: torch.optim.Adam(self.policy.parameters_model[key], config.learning_rate, eps=1e-5)
                          for key in self.model_keys}
        self.scheduler = {key: torch.optim.lr_scheduler.LinearLR(self.optimizer[key], start_factor=1.0, end_factor=0.5,
                                                                 total_iters=self.config.running_steps)
                          for key in self.model_keys}
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        for key in self.model_keys:
            q_eval_a = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999

            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
                q_next_a = q_next[key].gather(-1, actions_next_greedy[key].unsqueeze(-1).long()).reshape(bs)
            else:
                q_next_a = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

            # calculate the loss function
            td_error = (q_eval_a - q_target.detach()) * agent_mask[key]
            loss = (td_error ** 2).sum() / agent_mask[key].sum()
            self.optimizer[key].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_model[key], self.grad_clip_norm)
            self.optimizer[key].step()
            if self.scheduler[key] is not None:
                self.scheduler[key].step()

            lr = self.optimizer[key].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate": lr,
                f"{key}/loss_Q": loss.item(),
                f"{key}/predictQ": q_eval_a.mean().item()
            })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info

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
        _, actions_greedy, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                                rnn_hidden=rnn_hidden)
        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, q_next_seq = self.policy.Qtarget(observation=obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

        for key in self.model_keys:
            q_eval_a = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next = q_next_seq[key][:, 1:]
            if self.use_actions_mask:
                q_next[avail_actions[key][:, 1:] == 0] = -9999999

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
            self.optimizer[key].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_model[key], self.grad_clip_norm)
            self.optimizer[key].step()
            if self.scheduler is not None:
                self.scheduler[key].step()

            lr = self.optimizer[key].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate": lr,
                f"{key}/loss_Q": loss.item(),
                f"{key}/predictQ": q_eval_a.mean().item()
            })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
