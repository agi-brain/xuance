"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        for key in self.model_keys:
            q_eval_a = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1))

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999

            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
                q_next_a = q_next[key].gather(-1, actions_next_greedy[key].unsqueeze(-1).long())
            else:
                q_next_a = q_next[key].max(dim=-1, keepdim=True).values

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
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled']
        IDs = sample_Tensor['agent_ids']
        seq_len = filled.shape[1]

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            filled = filled.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_agents, -1, -1)
            obs[key] = obs[key].reshape([bs_rnn, seq_len + 1, -1])
            if self.use_actions_mask:
                avail_actions[key] = avail_actions[key].reshape(bs_rnn, seq_len + 1, -1)
            IDs = IDs.reshape(bs_rnn, seq_len + 1, -1)
            rnn_hidden = {key: self.policy.representation[key].init_hidden(bs_rnn)}
            _, actions_greedy, q_eval = self.policy(observation=obs,
                                                    agent_ids=IDs,
                                                    avail_actions=avail_actions,
                                                    rnn_hidden=rnn_hidden)
            target_rnn_hidden = {key: self.policy.target_representation[key].init_hidden(bs_rnn)}
            _, q_next_seq = self.policy.Qtarget(observation=obs,
                                                agent_ids=IDs,
                                                rnn_hidden=target_rnn_hidden)
            q_eval[key] = q_eval[key].reshape([batch_size, self.n_agents, seq_len + 1, -1])
            q_next_seq[key] = q_next_seq[key].reshape([batch_size, self.n_agents, seq_len + 1, -1])
            actions_greedy[key] = actions_greedy[key].reshape([batch_size, self.n_agents, -1])
        else:
            bs_rnn = batch_size
            rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
            _, actions_greedy, q_eval = self.policy(observation=obs,
                                                    agent_ids=IDs,
                                                    avail_actions=avail_actions,
                                                    rnn_hidden=rnn_hidden)
            target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
            _, q_next_seq = self.policy.Qtarget(observation=obs,
                                                agent_ids=IDs,
                                                rnn_hidden=target_rnn_hidden)

        for key in self.model_keys:
            if self.use_parameter_sharing:
                q_eval_a = q_eval[key][:, :, :-1].gather(-1, actions[key].long().unsqueeze(-1))
                q_next = q_next_seq[key][:, :, 1:]
                if self.use_actions_mask:
                    q_next[avail_actions[key][:, :, 1:] == 0] = -9999999
            else:
                q_eval_a = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1))
                q_next = q_next_seq[key][:, 1:]
                if self.use_actions_mask:
                    q_next[avail_actions[key][:, 1:] == 0] = -9999999

            if self.config.double_q:
                if self.use_parameter_sharing:
                    actions_next_greedy = actions_greedy[key].unsqueeze(-1)[:, :, 1:]
                else:
                    actions_next_greedy = actions_greedy[key].unsqueeze(-1)[:, 1:]
                q_next_a = q_next.gather(-1, actions_next_greedy.long().detach())
            else:
                q_next_a = q_next.max(dim=-1, keepdim=True).values

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
