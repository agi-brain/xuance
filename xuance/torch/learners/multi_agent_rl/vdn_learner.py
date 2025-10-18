"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class VDN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(VDN_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

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
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(dim=1).reshape(batch_size, 1)
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape(batch_size, 1)
        else:
            bs = batch_size
            rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
            terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        q_eval_a, q_next_a = {}, {}
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                             avail_actions=avail_actions, agent_key=key)
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().unsqueeze(-1)).reshape(bs)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

            q_eval_a[key] *= mask_values
            q_next_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next_a))

        q_tot_eval = self.policy.Q_tot(q_eval_a)
        q_tot_next = self.policy.Qtarget_tot(q_next_a)
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

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
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
            rewards_tot = rewards[key].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape([-1, 1])
        else:
            bs_rnn = batch_size
            rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=1).reshape(-1, 1)
            terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs_rnn=bs_rnn,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # calculate the individual Q values.
        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden)

        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.model_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().detach()).reshape(bs_rnn, seq_len)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs_rnn, seq_len)

            q_eval_a[key] = q_eval_a[key] * mask_values
            q_next_a[key] = q_next_a[key] * mask_values

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
                q_next_a[key] = q_next_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
            else:
                q_eval_a[key] = q_eval_a[key].reshape(-1, 1)
                q_next_a[key] = q_next_a[key].reshape(-1, 1)

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next_a, q_next=q_next))

        # calculate the total Q values.
        q_tot_eval = self.policy.Q_tot(q_eval_a)
        q_tot_next = self.policy.Qtarget_tot(q_next_a)
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

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target, td_errors=td_errors))

        return info
