"""
Weighted QMIX
Paper link: https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace


class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None):
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(WQMIX_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer,
                                            scheduler)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.optimizer = optimizer
        self.scheduler = scheduler

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

        # calculate Q_tot
        _, action_max, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        q_eval_centralized = self.policy.q_centralized(observation=obs, agent_ids=IDs)
        q_eval_next_centralized = self.policy.target_q_centralized(observation=obs_next, agent_ids=IDs)

        q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a, actions_next_greedy = {}, {}, {}, {}
        for key in self.model_keys:
            action_max[key] = action_max[key].unsqueeze(-1)
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)) * agent_mask[key]
            q_eval_centralized_a[key] = q_eval_centralized[key].gather(-1, action_max[key].long()) * agent_mask[key]

            if self.config.double_q:
                _, a_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                  avail_actions=avail_actions_next, agent_key=key)
                actions_next_greedy[key] = a_next_greedy[key].unsqueeze(-1)
            else:
                _, q_next_eval = self.policy.Qtarget(observation=obs_next, agent_ids=IDs, agent_key=key)
                if self.use_actions_mask:
                    q_next_eval[key][avail_actions_next[key] == 0] = -9999999
                actions_next_greedy[key] = q_next_eval[key].argmax(dim=-1, keepdim=True)

            q_eval_next_centralized_a[key] = q_eval_next_centralized[key].gather(
                -1, actions_next_greedy[key]) * agent_mask[key]

        q_tot_eval = self.policy.Q_tot(q_eval_a, state)  # calculate Q_tot
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state)  # calculate centralized Q
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state_next)  # y_i

        if self.use_parameter_sharing:
            rewards_tot = rewards[self.model_keys[0]].mean(dim=1)
            terminals_tot = terminals[self.model_keys[0]].all(dim=1, keepdim=False).float()
        else:
            rewards_tot = torch.concat([rewards[k] for k in self.model_keys], -1).mean(dim=-1, keepdim=True)
            terminals_tot = torch.concat([terminals[k] for k in self.model_keys], -1).all(dim=1, keepdim=True).float()

        target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.config.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

        # calculate losses and train
        loss_central = self.mse_loss(q_tot_centralized, target_value.detach())
        loss_qmix = (w.detach() * (td_error ** 2)).mean()
        loss = loss_qmix + loss_central
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
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

        # calculate Q_tot
        _, action_max, q_eval = self.policy(observation=obs, agent_ids=IDs,
                                            avail_actions=avail_actions, rnn_hidden=rnn_hidden)
        q_eval_centralized = self.policy.q_centralized(observation=obs, agent_ids=IDs, rnn_hidden=rnn_hidden)
        q_eval_next_centralized = self.policy.target_q_centralized(observation=obs, agent_ids=IDs,
                                                                   rnn_hidden=rnn_hidden)

        q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a = {}, {}, {}
        for key in self.model_keys:
            act_greedy = action_max[key][:, :-1].unsqueeze(-1)
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)) * agent_mask[key]
            q_eval_centralized_a[key] = q_eval_centralized[key][:, :-1].gather(-1, act_greedy.long()) * agent_mask[key]

            if self.config.double_q:
                act_next_greedy = action_max[key][:, 1:].unsqueeze(-1)
            else:
                _, q_next_seq = self.policy.Qtarget(observation=obs, agent_ids=IDs,
                                                    agent_key=key, rnn_hidden=target_rnn_hidden)
                q_next_eval = q_next_seq[key][:, 1:]
                if self.use_actions_mask:
                    q_next_eval[avail_actions[key][:, 1:] == 0] = -9999999
                act_next_greedy = q_next_eval.argmax(dim=-1, keepdim=True)

            q_eval_next_centralized_a[key] = q_eval_next_centralized[key].gather(-1, act_next_greedy) * agent_mask[key]

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(
                    batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_eval_centralized_a[key] = q_eval_centralized_a[key].reshape(
                    batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents, 1)
                q_eval_next_centralized_a[key] = q_eval_next_centralized_a[key].reshape(
                    batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents, 1)

        state_input = state[:, :-1].reshape([batch_size * seq_len, -1])
        state_input_next = state[:, 1:].reshape([batch_size * seq_len, -1])
        q_tot_eval = self.policy.Q_tot(q_eval_a, state_input)  # calculate Q_tot
        q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state_input)  # calculate centralized Q
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state_input_next)  # y_i

        if self.use_parameter_sharing:
            rewards_tot = rewards[self.model_keys[0]].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[self.model_keys[0]].all(dim=1, keepdim=False).float().reshape([-1, 1])
        else:
            rewards_tot = torch.concat([rewards[k] for k in self.model_keys], -1).mean(-1, True).reshape([-1, 1])
            terminals_tot = torch.concat([terminals[k]
                                          for k in self.model_keys], -1).all(-1, True).reshape([-1, 1]).float()

        target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.config.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

        # calculate losses and train
        loss_central = (((q_tot_centralized - target_value.detach()) ** 2) * filled).sum() / filled.sum()
        loss_qmix = (w.detach() * (td_error ** 2) * filled).sum() / filled.sum()
        loss = loss_qmix + loss_central
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        return info
