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
        self.mse_loss = nn.MSELoss()
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
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        # feedforward
        q_eval, q_next, actions_next = self._forward_transitions(batch)

        # calculate losses and update networks for each group of agents
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            rewards = batch.rewards[group].reshape(-1)
            terminals = batch.terminals[group].reshape(-1)

            q_eval_taken = q_eval[group].gather(-1, batch.actions[group].long().unsqueeze(-1)).reshape(-1)

            if self.config.double_q:
                q_next_taken = q_next[group].gather(-1, actions_next[group].long().unsqueeze(-1)).reshape(-1)
            else:
                q_next_taken = q_next[group].max(dim=-1, keepdim=True).values.reshape(-1)

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
