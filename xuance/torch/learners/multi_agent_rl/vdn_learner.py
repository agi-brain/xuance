"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
from argparse import Namespace
from xuance.common import AgentGrouping

import torch
from xuance.torch import Module
from xuance.torch.learners import OffPolicyMultiAgentLearner


class VDN_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(VDN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
        )

        rewards_tot = torch.stack([r for r in batch.rewards.agent_wise.values()], dim=1).mean(dim=1)
        terminals_tot = torch.stack([d for d in batch.terminals.agent_wise.values()], dim=1).all(dim=1).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # initialize rnn hidden states when use rnn
        rnn_states = self.model.init_rnn_states(batch.batch_size)

        # calculate the individual Q values
        model_output = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states
        )
        q_eval = model_output.values  # the individual Q values

        # calculate output with target networks
        with torch.no_grad():
            if self.use_rnn:
                actions_next = model_output.actions

                q_next = self.model.Qtarget(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states
                ).values
                q_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in q_eval.grouped_tensor.items()}
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                actions_next.grouped_tensor = {k: v[:, :, 1:] for k, v in actions_next.grouped_tensor.items()}

            else:
                q_next = self.model.Qtarget(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices,
                ).values

                if self.config.double_q:
                    actions_next = self.model(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices,
                                              avail_actions=batch.next_avail_actions).actions
                else:
                    actions_next = None

        # calculate target values
        q_eval_a, q_next_a = {}, {}
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape([batch.batch_size, n_agents, batch.seq_length])

            actions_taken = batch.actions.group(group)
            q_eval_taken = q_eval.group(group).gather(-1, actions_taken.long().unsqueeze(-1)).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

            if self.use_actions_mask:
                if self.use_rnn:
                    next_avail_actions = batch.avail_actions.group(group)[:, 1:]
                else:
                    next_avail_actions = batch.next_avail_actions.group(group)
                q_next.group(group)[next_avail_actions == 0] = -1e10

            if self.config.double_q:
                actions_next_taken = actions_next.group(group)
                q_next_taken = q_next.group(group).gather(-1, actions_next_taken.long().unsqueeze(-1)).reshape(
                    [batch.batch_size, n_agents, batch.seq_length])
            else:
                q_next_taken = q_next.group(group).max(dim=-1, keepdim=True).values.reshape(
                    [batch.batch_size, n_agents, batch.seq_length])

            q_eval_taken *= mask_values
            q_next_taken *= mask_values

            # get agent-wise values
            for i, agent_key in enumerate(self.groups[group]):
                q_eval_a[agent_key] = q_eval_taken[:, i]
                q_next_a[agent_key] = q_next_taken[:, i]

        q_tot_eval = self.model.Q_tot(q_eval_a).reshape(-1)
        q_tot_next = self.model.Qtarget_tot(q_next_a).reshape(-1)
        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            td_errors = (q_tot_eval - q_tot_target.detach()) * filled
            loss = (td_errors ** 2).sum() / filled.sum()
        else:
            loss = self.mse_loss(q_tot_eval, q_tot_target.detach())

        # update the networks
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

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_next=q_tot_next,
                                                q_tot_target=q_tot_target))

        return info
