"""
Weighted QMIX
Paper link: https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from argparse import Namespace
from xuance.common import AgentGrouping
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
        return_tensors = {}

        # calculate the individual Q value
        rnn_states = self.model.init_rnn_states(batch.batch_size)
        model_output = self.model(
            observations=batch.observations,
            agent_indices=batch.agent_indices,
            avail_actions=batch.avail_actions,
            rnn_states=rnn_states
        )
        actions_greedy, q_eval = model_output.actions, model_output.values
        if not self.agent_grouping.full_independent:
            actions_greedy = self.packed_tensor(actions_greedy)

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
                    q_next_seq = self.model.Qtarget(
                        observations=batch.observations,
                        agent_indices=batch.agent_indices,
                        rnn_states=rnn_states
                    ).values
                    return_tensors["q_next"] = {k: v[:, 1:] for k, v in q_next_seq.items()}

                q_eval = {k: v[:, :-1] for k, v in q_eval.items()}
                q_eval_centralized = {k: v[:, :-1] for k, v in q_eval_centralized.items()}
                q_next_centralized = {k: v[:, 1:] for k, v in q_next_centralized.items()}

                return_tensors["actions_greedy"] = {k: v[:, :-1] for k, v in actions_greedy.items()}
                return_tensors["next_actions_greedy"] = {k: v[:, 1:] for k, v in actions_greedy.items()}

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
                    if not self.agent_grouping.full_independent:
                        a_next_greedy = self.packed_tensor(a_next_greedy)
                    return_tensors["next_actions_greedy"] = a_next_greedy

                else:
                    q_next = self.model.Qtarget(
                        observations=batch.next_observations,
                        agent_indices=batch.agent_indices
                    ).values

                    if self.use_actions_mask:
                        for group in self.group_keys:
                            q_next[batch.next_avail_actions[group] == 0] = -1e10

                    return_tensors["q_next"] = q_next

                return_tensors["actions_greedy"] = actions_greedy

        return_tensors["q_eval"] = q_eval
        return_tensors["q_eval_centralized"] = q_eval_centralized
        return_tensors["q_next_centralized"] = q_next_centralized

        return return_tensors

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_parameter_sharing=self.use_parameter_sharing,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True,
                                         use_shared_rewards=True)

        rewards_tot = torch.stack([r for r in batch.rewards.values()], dim=1).sum(dim=1, keepdim=False)
        terminals_tot = torch.stack([d for d in batch.terminals.values()], dim=1).all(dim=1, keepdim=False).float()

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        # feedforward
        values_dict = self._forward_transitions(batch)
        q_eval = values_dict["q_eval"]
        q_eval_centralized = values_dict["q_eval_centralized"]
        q_next_centralized = values_dict["q_next_centralized"]
        actions_greedy = values_dict["actions_greedy"]

        # calculate target values
        q_eval_a, q_eval_centralized_a, q_next_centralized_a = {}, {}, {}
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape([-1, batch.seq_length])

            q_eval_taken = q_eval[group].gather(
                -1, batch.actions[group].long().unsqueeze(-1)).reshape([-1, batch.seq_length])
            q_eval_centralized_taken = q_eval_centralized[group].gather(
                -1, actions_greedy[group].long().unsqueeze(-1)).reshape(-1, batch.seq_length)

            if self.config.double_q:
                actions_next = values_dict["next_actions_greedy"][group]
            else:
                actions_next = values_dict["q_next"].argmax(dim=-1, keepdim=True)

            q_next_centralized_taken = q_next_centralized[group].gather(
                -1, actions_next.long().unsqueeze(-1)).reshape(-1, batch.seq_length)

            q_eval_taken *= mask_values
            q_eval_centralized_taken *= mask_values
            q_next_centralized_taken *= mask_values

            # get agent-wise values
            for i, agent_key in enumerate(self.groups[group]):
                q_eval_a[agent_key] = q_eval_taken.reshape([batch.batch_size, n_agents, batch.seq_length])[:, i]
                q_eval_centralized_a[agent_key] = q_eval_centralized_taken.reshape(
                    [batch.batch_size, n_agents, batch.seq_length])[:, i]
                q_next_centralized_a[agent_key] = q_next_centralized_taken.reshape(
                    [batch.batch_size, n_agents, batch.seq_length])[:, i]

        if self.use_rnn:
            state_input = batch.global_states[:, :-1].reshape([batch.batch_size * batch.seq_length, -1])
            state_input_next = batch.global_states[:, 1:].reshape([batch.batch_size * batch.seq_length, -1])
        else:
            state_input = batch.global_states
            state_input_next = batch.next_global_states
        # calculate Q_tot
        q_tot_eval = self.model.Q_tot(individual_values=q_eval_a,
                                      states=state_input).reshape(-1)
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
            condition_1 = ((actions_greedy == batch.actions.reshape(
                [-1, self.n_agents, 1])) * batch.agent_masks).all(dim=1)
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
