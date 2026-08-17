"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
from xuance.torch.learners import OffPolicyMultiAgentLearner
from xuance.torch.rl_models.modules import OffPolicyMARLBatch


class IQL_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(IQL_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def build_optimizer(self):
        self.optimizer = {
            key: torch.optim.Adam(self.model.individual_q_networks[key].parameters(),
                                  self.learning_rate, eps=1e-5)
            for key in self.group_keys
        }
        self.scheduler = {
            key: torch.optim.lr_scheduler.LinearLR(self.optimizer[key],
                                                   start_factor=1.0,
                                                   end_factor=self.end_factor_lr_decay,
                                                   total_iters=self.total_iters)
            for key in self.group_keys
        }

    def _forward_transitions(self, batch: OffPolicyMARLBatch):
        # calculate the individual Q values
        rnn_states = self.model.init_rnn_states(batch.batch_size)

        model_output = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states
        )
        q_eval = model_output.values

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

        for group in self.group_keys:
            if self.use_actions_mask:
                if self.use_rnn:
                    next_avail_actions = batch.avail_actions.group(group)[:, 1:]
                else:
                    next_avail_actions = batch.next_avail_actions.group(group)
                q_next.group(group)[next_avail_actions == 0] = -1e10

        return q_eval, q_next, actions_next

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # feedforward
        q_eval, q_next, actions_next = self._forward_transitions(batch)

        # calculate losses and update networks for each group of agents
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)

            actions_taken = batch.actions.packed(group)
            q_eval_taken = q_eval.packed(group).gather(-1, actions_taken.long().unsqueeze(-1)).reshape(-1)

            if self.config.double_q:
                actions_next_taken = actions_next.packed(group)
                q_next_taken = q_next.packed(group).gather(-1, actions_next_taken.long().unsqueeze(-1)).reshape(-1)
            else:
                q_next_taken = q_next.packed(group).max(dim=-1, keepdim=True).values.reshape(-1)

            q_target = rewards + (1 - terminals) * self.gamma * q_next_taken

            # calculate the loss function
            td_error = (q_eval_taken - q_target.detach()) * mask_values
            loss = (td_error ** 2).sum() / mask_values.sum()

            # update networks (backpropogation)
            self.optimizer[group].zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.individual_q_networks[group].parameters(),
                                               self.grad_clip_norm)
            self.optimizer[group].step()
            if self.scheduler is not None:
                self.scheduler[group].step()

            lr = self.optimizer[group].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{group}/learning_rate": lr,
                f"{group}/loss_Q": loss.item(),
                f"{group}/predictQ": q_eval_taken.mean().item()
            })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
