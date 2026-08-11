"""
Weighted QMIX
Paper link: https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from xuance.common import AgentGrouping
from argparse import Namespace
from operator import itemgetter


class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(WQMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.optimizer = torch.optim.Adam(self.model.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.total_iters)
        self.alpha = config.alpha
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)
            obs_next = self.packed_tensor(obs_next)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=-1, keepdim=True)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()

        info = self.callback.on_update_start(self.iterations, method="update", model=self.model,
                                             sample_Tensor=sample_Tensor,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # calculate Q_tot
        model_output = self.model(observations=obs,
                                  agent_indices=agent_indices,
                                  avail_actions=avail_actions)
        action_max, q_eval = model_output.actions, model_output.values

        q_eval_centralized = self.model.q_centralized(observations=obs,
                                                      agent_indices=agent_indices).values
        q_eval_next_centralized = self.model.target_q_centralized(observations=obs_next,
                                                                  agent_indices=agent_indices).values

        if self.config.double_q:
            a_next_greedy = self.model(observations=obs_next,
                                       agent_indices=agent_indices,
                                       avail_actions=avail_actions_next).actions
            q_next_eval = None
        else:
            q_next_eval = self.model.Qtarget(observations=obs_next,
                                             agent_indices=agent_indices).values
            a_next_greedy = None

        q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a, act_next = {}, {}, {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            action_max[key] = action_max[key].unsqueeze(-1)
            q_eval_a[key] = q_eval[key].gather(-1, actions[key].long().unsqueeze(-1)).reshape(batch_size)
            q_eval_centralized_a[key] = q_eval_centralized[key].gather(-1, action_max[key].long()).reshape(batch_size)

            if self.config.double_q:
                act_next[key] = a_next_greedy[key].unsqueeze(-1)
            else:
                if self.use_actions_mask:
                    q_next_eval[key][avail_actions_next[key] == 0] = -1e10
                act_next[key] = q_next_eval[key].argmax(dim=-1, keepdim=True)
            q_eval_next_centralized_a[key] = q_eval_next_centralized[key].gather(-1, act_next[key]).reshape(batch_size)

            q_eval_a[key] *= mask_values
            q_eval_centralized_a[key] *= mask_values
            q_eval_next_centralized_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_eval_centralized_a=q_eval_centralized_a,
                                                           act_next=act_next,
                                                           q_eval_next_centralized_a=q_eval_next_centralized_a))

        q_tot_eval = self.model.Q_tot(q_eval_a, state)  # calculate Q_tot
        q_tot_centralized = self.model.q_feedforward(q_eval_centralized_a, state)  # calculate centralized Q
        q_tot_next_centralized = self.model.target_q_feedforward(q_eval_next_centralized_a, state_next)  # y_i

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_centralized=q_tot_centralized,
                                                q_tot_next_centralized=q_tot_next_centralized,
                                                target_value=target_value, td_error=td_error, ones=ones, w=w))

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
        agent_indices = sample_Tensor['agent_indices']

        if not self.agent_grouping.full_independent:
            obs = self.packed_tensor(obs)

        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=1).reshape(-1, 1)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()

        info = self.callback.on_update_start(self.iterations, method="update_rnn", model=self.model,
                                             sample_Tensor=sample_Tensor,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # calculate Q_tot
        rnn_states = self.model.init_rnn_states(batch_size)
        model_output = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions,
                                  rnn_states=rnn_states)
        actions_greedy, q_eval = model_output.actions, model_output.values

        rnn_states_cent = self.model.init_centralized_rnn_states(batch_size)
        q_eval_centralized = self.model.q_centralized(observations=obs, agent_indices=agent_indices,
                                                      rnn_states=rnn_states_cent).values
        q_eval_next_centralized = self.model.target_q_centralized(observations=obs, agent_indices=agent_indices,
                                                                  rnn_states=rnn_states_cent).values

        if self.config.double_q:
            q_next_seq = None
        else:
            q_next_seq = self.model.Qtarget(observations=obs, agent_indices=agent_indices, rnn_states=rnn_states).values

        q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a = {}, {}, {}
        for key in self.agent_keys:
            mask_values = agent_mask[key]
            act_greedy = actions_greedy[key][:, :-1].unsqueeze(-1)
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(
                batch_size, seq_len)
            q_eval_centralized_a[key] = q_eval_centralized[key][:, :-1].gather(-1, act_greedy.long()).reshape(
                batch_size, seq_len)

            if self.config.double_q:
                act_next = actions_greedy[key][:, 1:].unsqueeze(-1)
            else:
                q_next_eval = q_next_seq[key][:, 1:]
                if self.use_actions_mask:
                    q_next_eval[avail_actions[key][:, 1:] == 0] = -1e10
                act_next = q_next_eval.argmax(dim=-1, keepdim=True)
            q_eval_next_centralized_a[key] = q_eval_next_centralized[key][:, 1:].gather(-1, act_next).reshape(
                batch_size, seq_len)

            q_eval_a[key] *= mask_values
            q_eval_centralized_a[key] *= mask_values
            q_eval_next_centralized_a[key] *= mask_values

            info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_rnn",
                                                           mask_values=mask_values, q_eval_a=q_eval_a,
                                                           q_eval_centralized_a=q_eval_centralized_a,
                                                           act_next=act_next,
                                                           q_eval_next_centralized_a=q_eval_next_centralized_a))

        state_input = state[:, :-1].reshape([batch_size * seq_len, -1])
        state_input_next = state[:, 1:].reshape([batch_size * seq_len, -1])
        q_tot_eval = self.model.Q_tot(q_eval_a, state_input)  # calculate Q_tot
        q_tot_centralized = self.model.q_feedforward(q_eval_centralized_a, state_input)  # calculate centralized Q
        q_tot_next_centralized = self.model.target_q_feedforward(q_eval_next_centralized_a, state_input_next)  # y_i

        target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1 = ((act_greedy == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Qmix": loss_qmix.item(),
            "loss_central": loss_central.item(),
            "loss": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_centralized=q_tot_centralized,
                                                q_tot_next_centralized=q_tot_next_centralized,
                                                target_value=target_value, td_error=td_error, ones=ones, w=w))

        return info
