"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
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
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

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
        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        q_eval_a, q_next_a = {}, {}
        for key in self.model_keys:
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)) * agent_mask[key]
            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -9999999
            if self.config.double_q:
                _, actions_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                        avail_actions=avail_actions, agent_key=key)
                q_next_a[key] = q_next[key].gather(-1, actions_next_greedy[key].long().unsqueeze(-1)) * agent_mask[key]
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values * agent_mask[key]

        q_tot_eval = self.policy.Q_tot(q_eval_a)
        q_tot_next = self.policy.Qtarget_tot(q_next_a)

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
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        seq_len = sample['sequence_length']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])
        IDs = sample_Tensor['agent_ids']

        rnn_hidden, target_rnn_hidden = {}, {}
        for key in self.model_keys:
            if self.use_parameter_sharing:
                obs[key] = obs[key].reshape([bs_rnn, seq_len + 1, -1])
                actions[key] = actions[key].reshape([bs_rnn, seq_len])
                agent_mask[key] = agent_mask[key].reshape([bs_rnn, seq_len, 1])
                if self.use_actions_mask:
                    avail_actions[key] = avail_actions[key].reshape(bs_rnn, seq_len + 1, -1)
                IDs = IDs.reshape(bs_rnn, seq_len + 1, -1)

            rnn_hidden[key] = self.policy.representation[key].init_hidden(bs_rnn)
            target_rnn_hidden[key] = self.policy.target_representation[key].init_hidden(bs_rnn)

        # calculate the individual Q values.
        _, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden)
        _, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.model_keys:
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)) * agent_mask[key]

            q_next[key] = q_next_seq[key][:, 1:]
            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -9999999
            if self.config.double_q:
                actions_next_greedy = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a[key] = q_next[key].gather(-1, actions_next_greedy[key].long().detach()) * agent_mask[key]
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values * agent_mask[key]

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_next_a[key] = q_next_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents, 1)

        # calculate the total Q values.
        q_tot_eval = self.policy.Q_tot(q_eval_a)
        q_tot_next = self.policy.Qtarget_tot(q_next_a)

        if self.use_parameter_sharing:
            rewards_tot = rewards[self.model_keys[0]].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[self.model_keys[0]].all(dim=1, keepdim=False).float().reshape([-1, 1])
        else:
            rewards_tot = torch.concat([rewards[k] for k in self.model_keys], -1).mean(-1, True).reshape([-1, 1])
            terminals_tot = torch.concat([terminals[k] for k in self.model_keys], -1).all(-1, True).reshape([-1, 1]).float()

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
