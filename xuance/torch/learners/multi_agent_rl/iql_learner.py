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
        self.mse_loss = nn.MSELoss()
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.use_actions_mask = config.use_actions_mask
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.optimizer = optimizer
        self.scheduler = scheduler

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample,
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

        for key in self.model_keys:
            _, _, q_eval = self.policy(obs, IDs, avail_actions, agent_key=key)
            if self.use_parameter_sharing:
                actions_key = actions[key].long().reshape([self.config.batch_size, self.n_agents, 1])
            else:
                actions_key = actions[key].long().reshape([self.config.batch_size, 1])
            q_eval_a = q_eval[key].gather(-1, actions_key)
            _, q_next = self.policy.Qtarget(obs_next, IDs, agent_key=key)

            if self.use_actions_mask:
                q_next[avail_actions_next == 0] = -9999999

            if self.config.double_q:
                _, action_next_greedy, q_next_eval = self.policy(obs_next, IDs, agent_key=key)
                q_next_a = q_next[key].gather(-1, action_next_greedy[key].unsqueeze(-1).long())
            else:
                q_next_a = q_next[key].max(dim=-1, keepdim=True).values

            q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

            # calculate the loss function
            td_error = (q_eval_a - q_target.detach()) * agent_mask[key]
            loss = (td_error ** 2).sum() / agent_mask[key].sum()
            self.optimizer[key].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
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
        sample_Tensor = self.build_training_data(sample,
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

        # Current Q
        rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, actions_greedy, q_eval = self.policy(obs.reshape(-1, self.episode_length + 1, self.dim_obs),
                                                IDs.reshape(-1, self.episode_length + 1, self.n_agents),
                                                *rnn_hidden,
                                                avail_actions=avail_actions.reshape(-1, self.episode_length + 1,
                                                                                    self.dim_act))
        q_eval = q_eval[:, :-1].reshape(batch_size, self.n_agents, self.episode_length, self.dim_act)
        actions_greedy = actions_greedy.reshape(batch_size, self.n_agents, self.episode_length + 1, 1)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, self.episode_length, 1]))

        # Target Q
        target_rnn_hidden = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
        _, q_next = self.policy.target_Q(obs.reshape(-1, self.episode_length + 1, self.dim_obs),
                                         IDs.reshape(-1, self.episode_length + 1, self.n_agents),
                                         *target_rnn_hidden)
        q_next = q_next[:, 1:].reshape(batch_size, self.n_agents, self.episode_length, self.dim_act)
        q_next[avail_actions[:, :, 1:] == 0] = -9999999

        # use double-q trick
        if self.args.double_q:
            action_next_greedy = actions_greedy[:, :, 1:]
            q_next_a = q_next.gather(-1, action_next_greedy.long().detach())
        else:
            q_next_a = q_next.max(dim=-1, keepdim=True).values

        filled_n = filled.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        rewards = rewards.expand(-1, self.n_agents, -1, -1)
        terminals = terminals.unsqueeze(1).expand(batch_size, self.n_agents, self.episode_length, 1)
        q_target = rewards + (1 - terminals) * self.gamma * q_next_a

        # calculate the loss function
        td_errors = q_eval_a - q_target.detach()
        td_errors *= filled_n
        loss = (td_errors ** 2).sum() / filled_n.sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_eval_a.mean().item()
        }

        return info
