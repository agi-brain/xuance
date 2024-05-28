"""
Value Decomposition Networks (VDN)
Paper link:
https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class VDN_Learner(LearnerMAS):
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
        super(VDN_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.use_actions_mask = config.use_actions_mask
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.optimizer = optimizer
        self.scheduler = scheduler

    def update(self, sample):
        self.iterations += 1

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
        if self.use_parameter_sharing:
            q_eval_a = q_eval[self.model_keys[0]].gather(-1, actions[self.model_keys[0]].long().unsqueeze(-1))
        else:
            q_eval_a = [q_eval[k].gather(-1, actions[k].long().unsqueeze(-1)) for k in self.model_keys]
            q_eval_a = torch.concat(q_eval_a, dim=None)
        q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask)

        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)
        if self.use_actions_mask:
            for key in self.model_keys:
                q_next[key][avail_actions_next[key] == 0] = -9999999
        if self.config.double_q:
            _, actions_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs, avail_actions=avail_actions)
            q_next_a = [q_next[k].gather(-1, actions_next_greedy[k].long().unsqueeze(-1)) for k in self.model_keys]
            q_next_a = torch.concat(q_next_a, dim=None)
        else:
            q_next_a = [q_next[k].max(dim=-1, keepdim=True) for k in self.model_keys]
        q_tot_next = self.policy.Qtarget_tot(q_next_a * agent_mask)
        q_tot_target = rewards + (1 - terminals) * self.gamma * q_tot_next

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
            filled = filled.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
            obs[key] = obs[key].reshape([bs_rnn, seq_len + 1, -1])
            avail_actions_input = {key: avail_actions[key].reshape(bs_rnn, seq_len + 1, -1)}
            IDs = IDs.reshape(bs_rnn, seq_len + 1, -1)
            rnn_hidden = {key: self.policy.representation[key].init_hidden(bs_rnn)}
            _, actions_greedy, q_eval = self.policy(observation=obs,
                                                    agent_ids=IDs,
                                                    avail_actions=avail_actions_input,
                                                    rnn_hidden=rnn_hidden)
            target_rnn_hidden = {key: self.policy.target_representation[key].init_hidden(bs_rnn)}
            _, q_next_seq = self.policy.Qtarget(observation=obs,
                                                agent_ids=IDs,
                                                rnn_hidden=target_rnn_hidden)
            q_eval[key] = q_eval[key].reshape([batch_size, self.n_agents, seq_len + 1, -1])
            q_next_seq[key] = q_next_seq[key].reshape([batch_size, self.n_agents, seq_len + 1, -1])
            actions_greedy[key] = actions_greedy[key].reshape([batch_size, self.n_agents, -1])
        else:
            bs_rnn = batch_size * self.n_agents
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
