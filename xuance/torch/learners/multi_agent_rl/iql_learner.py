"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping
from argparse import Namespace


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IQL_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = {key: torch.optim.Adam(self.model.individual_q_networks[key].parameters(),
                                                config.learning_rate, eps=1e-5)
                          for key in self.group_keys}
        self.scheduler = {key: torch.optim.lr_scheduler.LinearLR(self.optimizer[key],
                                                                 start_factor=1.0,
                                                                 end_factor=self.end_factor_lr_decay,
                                                                 total_iters=self.total_iters)
                          for key in self.group_keys}
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def update(self, sample):
        self.iterations += 1

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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            obs_next = self.packed_tensor(obs_next)
            agent_mask = self.packed_tensor(agent_mask)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor, batch_size=batch_size)

        q_eval = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions).values
        q_next = self.model.Qtarget(observations=obs_next, agent_indices=agent_indices).values

        if self.config.double_q:
            actions_next = self.model(observations=obs_next, agent_indices=agent_indices,
                                      avail_actions=avail_actions_next).actions
            if not self.agent_grouping.full_independent:
                actions_next = self.packed_tensor(actions_next)

        else:
            actions_next = None

        for group, agent_keys in self.groups.items():
            n_agents = self.n_group_agents[group]
            bs = batch_size * n_agents
            mask_values = agent_mask[group]
            q_eval_a = q_eval[group].gather(-1, actions[group].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[group][avail_actions_next[group] == 0] = -1e10

            if self.config.double_q:
                q_next_a = q_next[group].gather(-1, actions_next[group].unsqueeze(-1).long()).reshape(bs)
            else:
                q_next_a = q_next[group].max(dim=-1, keepdim=True).values.reshape(bs)

            q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_a

            # calculate the loss function
            td_error = (q_eval_a - q_target.detach()) * mask_values
            loss = (td_error ** 2).sum() / mask_values.sum()

            self.optimizer[group].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.individual_q_networks[group].parameters(),
                                               self.grad_clip_norm)
            self.optimizer[group].step()
            if self.scheduler is not None:
                self.scheduler[group].step()

            lr = self.optimizer[group].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate": lr,
                f"{group}/loss_Q": loss.item(),
                f"{group}/predictQ": q_eval_a.mean().item()
            })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            actions = self.packed_tensor(actions)
            rewards = self.packed_tensor(rewards)
            terminals = self.packed_tensor(terminals)

            agent_mask = self.packed_tensor(agent_mask)

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor)

        # calculate the individual Q values.
        rnn_states = self.model.init_rnn_states(batch_size)
        model_output = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy, q_eval = model_output.actions, model_output.values
        if not self.agent_grouping.full_independent:
            actions_greedy = self.packed_tensor(actions_greedy)

        q_next_seq = self.model.Qtarget(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states).values

        for group, agent_keys in self.groups.items():
            n_agents = len(agent_keys)
            bs = batch_size * n_agents
            group_filled = filled.unsqueeze(1).expand(batch_size, n_agents, seq_len).reshape([bs, seq_len])
            mask_values = agent_mask[group] * group_filled
            q_eval_a = q_eval[group][:, :-1].gather(-1, actions[group].long().unsqueeze(-1)).reshape(bs, seq_len)
            q_next = q_next_seq[group][:, 1:]

            if self.use_actions_mask:
                q_next[avail_actions[group][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = actions_greedy[group][:, 1:].unsqueeze(-1)
                q_next_a = q_next.gather(-1, act_next.long().detach()).reshape(bs, seq_len)
            else:
                q_next_a = q_next.max(dim=-1, keepdim=True).values.reshape(bs, seq_len)

            q_target = rewards[group] + (1 - terminals[group]) * self.gamma * q_next_a

            # calculate the loss function
            td_errors = (q_eval_a - q_target.detach()) * mask_values
            loss = (td_errors ** 2).sum() / mask_values.sum()

            self.optimizer[group].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.individual_q_networks[group].parameters(),
                                               self.grad_clip_norm)
            self.optimizer[group].step()
            if self.scheduler is not None:
                self.scheduler[group].step()

            lr = self.optimizer[group].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate": lr,
                f"{group}/loss_Q": loss.item(),
                f"{group}/predictQ": q_eval_a.mean().item()
            })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info))

        return info
