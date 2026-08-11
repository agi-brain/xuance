"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping
from argparse import Namespace
from operator import itemgetter


class VDN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(VDN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = torch.optim.Adam(self.model.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
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
            obs_next = self.packed_tensor(obs_next)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()

        info = self.callback.on_update_start(self.iterations, method="update",
                                             model=self.model, sample_Tensor=sample_Tensor,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        q_eval = self.model(observations=obs,
                            agent_indices=agent_indices,
                            avail_actions=avail_actions).values
        q_next = self.model.Qtarget(observations=obs_next,
                                    agent_indices=agent_indices).values

        if self.config.double_q:
            actions_next = self.model(observations=obs_next,
                                      agent_indices=agent_indices,
                                      avail_actions=avail_actions_next).actions
        else:
            actions_next = None

        q_eval_a, q_next_a = {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(batch_size)

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                q_next_a[key] = q_next[key].gather(-1, actions_next[key].long().unsqueeze(-1)).reshape(batch_size)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(batch_size)

            q_eval_a[key] *= mask_values
            q_next_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next_a))

        q_tot_eval = self.model.Q_tot(q_eval_a)
        q_tot_next = self.model.Qtarget_tot(q_next_a)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        loss = self.mse_loss(q_tot_eval, q_tot_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters_model, self.grad_clip_norm)
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
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info,
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=1).reshape(-1, 1)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             model=self.model, sample_Tensor=sample_Tensor,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # calculate the individual Q values.
        rnn_states = self.model.init_rnn_states(batch_size)
        model_output = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy, q_eval = model_output.actions, model_output.values

        q_next_seq = self.model.Qtarget(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states).values

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(batch_size, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = actions_greedy[key].unsqueeze(-1)[:, 1:]
                q_next_a[key] = q_next[key].gather(-1, act_next.long().detach()).reshape(batch_size, seq_len)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(batch_size, seq_len)

            q_eval_a[key] = q_eval_a[key] * mask_values
            q_next_a[key] = q_next_a[key] * mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_next_a=q_next_a, q_next=q_next))

        # calculate the total Q values.
        q_tot_eval = self.model.Q_tot(q_eval_a)
        q_tot_next = self.model.Qtarget_tot(q_next_a)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        td_errors = (q_tot_eval - q_tot_target.detach()) * filled
        loss = (td_errors ** 2).sum() / filled.sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters_model, self.grad_clip_norm)
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
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target, td_errors=td_errors))

        return info
