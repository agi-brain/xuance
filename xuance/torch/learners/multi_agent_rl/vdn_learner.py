"""
Value Decomposition Networks (VDN)
Paper link: https://arxiv.org/pdf/1706.05296.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class VDN_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(VDN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def _forward_transitions(self, batch: OffPolicyMARLBatch):
        # calculate the individual Q values.
        rnn_states = self.model.init_rnn_states(batch.batch_size)

        model_output = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states
        )
        actions_greedy = model_output.actions
        q_eval = model_output.values

        with torch.no_grad():
            if self.use_rnn:
                if not self.agent_grouping.full_independent:
                    actions_greedy = self.packed_tensor(actions_greedy)

                q_next_seq = self.model.Qtarget(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states
                ).values
                q_eval = {k: v[:, :-1] for k, v in q_eval.items()}
                q_next = {k: v[:, 1:] for k, v in q_next_seq.items()}
                actions_next = {k: v[:, 1:] for k, v in actions_greedy.items()}

            else:
                q_next = self.model.Qtarget(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states
                ).values
                if self.config.double_q:
                    actions_next = self.model(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices,
                                              avail_actions=batch.next_avail_actions).actions
                    if not self.agent_grouping.full_independent:
                        actions_next = self.packed_tensor(actions_next)
                else:
                    actions_next = None

        for group in self.group_keys:
            if self.use_actions_mask:
                if self.use_rnn:
                    next_avail_actions = batch.next_avail_actions
                else:
                    next_avail_actions = batch.avail_actions[group][:, 1:]
                q_next[group][next_avail_actions == 0] = -1e10

        return q_eval, q_next, actions_next

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask,
                                         use_shared_rewards=True)

        rewards_tot = torch.stack([r for r in batch.rewards.values()], dim=1).sum(dim=1, keepdim=False)
        terminals_tot = torch.stack([d for d in batch.terminals.values()], dim=1).all(dim=1, keepdim=False).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # feedforward
        q_eval, q_next, actions_next = self._forward_transitions(batch)

        # calculate losses and update networks for each group of agents
        q_eval_a, q_next_a = {}, {}
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape([-1, batch.seq_length])

            q_eval_taken = q_eval[group].gather(-1, batch.actions[group].long().unsqueeze(-1)).reshape(
                [-1, batch.seq_length])

            if self.config.double_q:
                q_next_taken = q_next[group].gather(-1, actions_next[group].long().unsqueeze(-1)).reshape(
                    [-1, batch.seq_length])
            else:
                q_next_taken = q_next[group].max(dim=-1, keepdim=True).values.reshape([-1, batch.seq_length])

            q_eval_taken *= mask_values
            q_next_taken *= mask_values

            for i, agent_key in enumerate(self.groups[group]):
                q_eval_a[agent_key] = q_eval_taken.reshape([batch.batch_size, n_agents, batch.seq_length])[:, i]
                q_next_a[agent_key] = q_next_taken.reshape([batch.batch_size, n_agents, batch.seq_length])[:, i]

        q_tot_eval = self.model.Q_tot(q_eval_a).reshape(-1)
        q_tot_next = self.model.Qtarget_tot(q_next_a).reshape(-1)
        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        if self.use_rnn:
            td_errors = (q_tot_eval - q_tot_target.detach()) * batch.filled_masks.reshape(-1)
            loss = (td_errors ** 2).sum() / batch.filled_masks.sum()
        else:
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
