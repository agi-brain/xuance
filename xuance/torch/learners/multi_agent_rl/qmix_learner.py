"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link: http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class QMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QMIX_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.use_actions_mask = config.use_actions_mask
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
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

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            q_eval_a = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)) * agent_mask[key]
            q_eval_a = q_eval_a.reshape([-1, self.n_agents, 1])

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999
            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                        avail_actions=avail_actions)
                q_next_a = q_next[key].gather(-1, actions_next_greedy[key].long().unsqueeze(-1)) * agent_mask[key]
            else:
                q_next_a = q_next[key].max(dim=-1, keepdim=True).values * agent_mask[key]
            q_next_a = q_next_a.reshape([-1, self.n_agents, 1])

        else:
            q_eval_a = torch.concat([q_eval[k].gather(-1, actions[k].long().unsqueeze(-1)) * agent_mask[k]
                                     for k in self.model_keys], dim=-1).reshape([-1, self.n_agents, 1])
            if self.use_actions_mask:
                for key in self.model_keys:
                    q_next[key][avail_actions_next[key] == 0] = -9999999
            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                        avail_actions=avail_actions)
                q_next_a_list = [q_next[k].gather(-1, actions_next_greedy[k].long().unsqueeze(-1)) * agent_mask[k]
                                 for k in self.model_keys]
            else:
                q_next_a_list = [q_next[k].max(dim=-1, keepdim=True).values * agent_mask[k] for k in self.model_keys]
            q_next_a = torch.concat(q_next_a_list, dim=-1).reshape([-1, self.n_agents, 1])

        q_tot_eval = self.policy.Q_tot(q_eval_a, state)
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state_next)

        if self.use_parameter_sharing:
            rewards_tot = rewards[self.model_keys[0]].mean(dim=1)
            terminals_tot = terminals[self.model_keys[0]].all(dim=1, keepdim=False).float()
        else:
            rewards_tot = torch.concat([rewards[k] for k in self.model_keys], -1).mean(dim=-1, keepdim=True)
            terminals_tot = torch.concat([terminals[k] for k in self.model_keys], -1).all(dim=1, keepdim=True).float()

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        loss = self.mse_loss(q_tot_eval, q_tot_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            obs[key] = obs[key].reshape([bs_rnn, seq_len + 1, -1])
            avail_actions_input = {key: avail_actions[key].reshape(bs_rnn, seq_len + 1, -1)}
            IDs = IDs.reshape(bs_rnn, seq_len + 1, -1)

            rnn_hidden = {key: self.policy.representation[key].init_hidden(bs_rnn)}
            _, actions_greedy, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions_input,
                                                    rnn_hidden=rnn_hidden)
            q_eval_a = q_eval[key][:, :-1].reshape([batch_size, self.n_agents, seq_len, -1]).gather(
                -1, actions[key].long().unsqueeze(-1)) * agent_mask[key]
            q_eval_a = q_eval_a.transpose(1, 2).reshape(-1, self.n_agents, 1)

            target_rnn_hidden = {key: self.policy.target_representation[key].init_hidden(bs_rnn)}
            _, q_next_seq = self.policy.Qtarget(observation=obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)
            q_next = q_next_seq[key][:, 1:].reshape([batch_size, self.n_agents, seq_len, -1]) * agent_mask[key]
            if self.use_actions_mask:
                q_next[avail_actions[key][:, :, 1:] == 0] = -9999999
            if self.config.double_q:
                act_next = actions_greedy[key].reshape([batch_size, self.n_agents, -1])[:, :, 1:].unsqueeze(-1).long()
                q_next_a = q_next.gather(-1, act_next.detach()).transpose(1, 2).reshape(-1, self.n_agents, 1)
            else:
                q_next_a = q_next.max(dim=-1, keepdim=True).values.transpose(1, 2).reshape(-1, self.n_agents, 1)

        else:
            bs_rnn = batch_size
            rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
            _, actions_greedy, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                                    rnn_hidden=rnn_hidden)
            q_eval_a = torch.concat([q_eval[k][:, :-1].gather(-1, actions[k].long().unsqueeze(-1)) * agent_mask[k]
                                     for k in self.model_keys], dim=-1).reshape(-1, self.n_agents, 1)

            target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
            _, q_next_seq = self.policy.Qtarget(observation=obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

            q_next = {k: q_next_seq[k][:, 1:] for k in self.model_keys}
            if self.use_actions_mask:
                for k in self.model_keys:
                    q_next[k][avail_actions[k][:, 1:] == 0] = -9999999

            if self.config.double_q:
                actions_next_greedy = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a_list = [q_next[k].gather(-1, actions_next_greedy[k].long().detach()) * agent_mask[k] for k in
                                 self.model_keys]
            else:
                q_next_a_list = [q_next[k].max(dim=-1, keepdim=True).values * agent_mask[k] for k in self.model_keys]
            q_next_a = torch.concat(q_next_a_list, dim=-1).reshape(-1, self.n_agents, 1)

        q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1].reshape([batch_size * seq_len, -1]))
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state[:, 1:].reshape([batch_size * seq_len, -1]))

        if self.use_parameter_sharing:
            rewards_tot = rewards[self.model_keys[0]].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[self.model_keys[0]].all(dim=1, keepdim=False).float().reshape([-1, 1])
        else:
            rewards_tot = torch.concat([rewards[k] for k in self.model_keys], -1).mean(-1, True).reshape([-1, 1])
            terminals_tot = torch.concat([terminals[k] for k in self.model_keys], -1).all(-1, True).reshape(
                [-1, 1]).float()

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        td_errors = (q_tot_eval - q_tot_target.detach()) * filled
        loss = (td_errors ** 2).sum() / filled.sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
