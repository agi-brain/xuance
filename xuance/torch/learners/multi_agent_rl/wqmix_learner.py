"""
Weighted QMIX
Paper link: https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.learners.multi_agent_rl.iql_learner import IQL_Learner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class WQMIX_Learner(IQL_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(WQMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.alpha = config.alpha

    def build_optimizer(self):
        super(IQL_Learner, self).build_optimizer()

    def _forward_transitions(self, batch: OffPolicyMARLBatch):
        # calculate the individual Q value
        rnn_states = self.model.init_rnn_states(batch.batch_size)
        model_output = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states
        )
        actions_greedy = model_output.actions
        q_eval = model_output.values

        rnn_states_cent = self.model.init_centralized_rnn_states(batch.batch_size)
        q_eval_centralized = self.model.q_centralized(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            rnn_states=rnn_states_cent
        ).values

        with torch.no_grad():
            if self.use_rnn:
                q_next_centralized = self.model.target_q_centralized(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states_cent
                ).values

                if not self.config.double_q:
                    q_next = self.model.Qtarget(
                        observations=batch.observations,
                        agent_indices=batch.agent_indices,
                        rnn_states=rnn_states
                    ).values
                    q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                else:
                    q_next = None

                q_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in q_eval.grouped_tensor.items()}
                q_eval_centralized.grouped_tensor = {k: v[:, :, :-1]
                                                     for k, v in q_eval_centralized.grouped_tensor.items()}
                q_next_centralized.grouped_tensor = {k: v[:, :, 1:]
                                                     for k, v in q_next_centralized.grouped_tensor.items()}
                next_actions_greedy = AgentGroupedTensor({k: v[:, :, 1:]
                                                          for k, v in actions_greedy.grouped_tensor.items()},
                                                         self.agent_grouping)
                actions_greedy.grouped_tensor = {k: v[:, :, :-1] for k, v in actions_greedy.grouped_tensor.items()}

            else:
                q_next_centralized = self.model.target_q_centralized(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices
                ).values

                if self.config.double_q:
                    a_next_greedy = self.model(
                        observations=batch.next_observations,
                        agent_indices=batch.agent_indices,
                        avail_actions=batch.next_avail_actions
                    ).actions
                    next_actions_greedy = a_next_greedy
                    q_next = None
                else:
                    q_next = self.model.Qtarget(
                        observations=batch.next_observations,
                        agent_indices=batch.agent_indices
                    ).values

                    if self.use_actions_mask:
                        for group in self.group_keys:
                            q_next[batch.next_avail_actions[group] == 0] = -1e10

                    next_actions_greedy = None

        q_eval_a, q_eval_centralized_a, q_next_centralized_a = {}, {}, {}
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape([batch.batch_size, n_agents, batch.seq_length])

            actions_taken = batch.actions.group(group).long().unsqueeze(-1)
            q_eval_taken = q_eval.group(group).gather(-1, actions_taken).reshape(
                [batch.batch_size, n_agents, batch.seq_length])
            actions_greedy_taken = actions_greedy.group(group).long().unsqueeze(-1)
            q_eval_centralized_taken = q_eval_centralized.group(group).gather(-1, actions_greedy_taken).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

            if self.config.double_q:
                actions_next_taken = next_actions_greedy.group(group).long().unsqueeze(-1)
            else:
                actions_next_taken = q_next.group(group).argmax(dim=-1, keepdim=True).long().unsqueeze(-1)

            q_next_centralized_taken = q_next_centralized.group(group).gather(-1, actions_next_taken).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

            q_eval_taken *= mask_values
            q_eval_centralized_taken *= mask_values
            q_next_centralized_taken *= mask_values

            # get agent-wise values
            for i, agent_key in enumerate(self.groups[group]):
                q_eval_a[agent_key] = q_eval_taken[:, i]
                q_eval_centralized_a[agent_key] = q_eval_centralized_taken[:, i]
                q_next_centralized_a[agent_key] = q_next_centralized_taken[:, i]

        return q_eval_a, q_eval_centralized_a, q_next_centralized_a, actions_greedy

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True
        )

        rewards_tot = torch.stack([r for r in batch.rewards.agent_wise.values()], dim=1).mean(dim=1)
        terminals_tot = torch.stack([d for d in batch.terminals.agent_wise.values()], dim=1).all(dim=1).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)
        # feedforward
        q_eval_a, q_eval_centralized_a, q_next_centralized_a, actions_greedy = self._forward_transitions(batch)

        if self.use_rnn:
            state_input = batch.global_states[:, :-1].reshape([batch.batch_size * batch.seq_length, -1])
            state_input_next = batch.global_states[:, 1:].reshape([batch.batch_size * batch.seq_length, -1])
        else:
            state_input = batch.global_states
            state_input_next = batch.next_global_states
        # calculate Q_tot
        q_tot_eval = self.model.Q_tot(individual_values=q_eval_a, states=state_input).reshape(-1)
        # calculate centralized Q
        q_tot_centralized = self.model.q_feedforward(individual_values=q_eval_centralized_a,
                                                     states=state_input).reshape(-1)
        # calculate y_i
        q_tot_next_centralized = self.model.target_q_feedforward(individual_values=q_next_centralized_a,
                                                                 states=state_input_next).reshape(-1)
        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)

        target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
        td_error = q_tot_eval - target_value.detach()

        # calculate the weights
        ones = torch.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1_list = []
            for group, agent_keys in self.groups.items():
                n_agents = self.n_group_agents[group]
                mask_values = batch.valid_mask(group, n_agents).reshape([batch.batch_size, n_agents, batch.seq_length])
                a_greedy = actions_greedy.group(group)
                act = batch.actions[group].reshape([batch.batch_size, n_agents, batch.seq_length])
                condition_1_list.append(((a_greedy == act) * mask_values))
            condition_1 = torch.concat(condition_1_list, dim=1).all(dim=1).reshape(-1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = torch.where(conditions, ones, w)
        elif self.config.agent == "OWQMIX":
            condition = td_error < 0
            w = torch.where(condition, ones, w)
        else:
            raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            loss_central = (((q_tot_centralized - target_value.detach()) ** 2) * filled).sum() / filled.sum()
            loss_qmix = (w.detach() * (td_error ** 2) * filled).sum() / filled.sum()
        else:
            loss_central = self.mse_loss(q_tot_centralized, target_value.detach())
            loss_qmix = (w.detach() * (td_error ** 2)).mean()

        loss = loss_qmix + loss_central

        # update the networks
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

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                q_tot_eval=q_tot_eval, q_tot_centralized=q_tot_centralized,
                                                q_tot_next_centralized=q_tot_next_centralized,
                                                target_value=target_value, td_error=td_error, ones=ones, w=w))

        return info
