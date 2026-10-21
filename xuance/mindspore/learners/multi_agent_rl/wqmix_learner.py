"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, Module, Tensor, ops
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch, AgentGroupedTensor
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class WQMIX_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(WQMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        self.alpha = config.alpha
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()

    def forward_fn(self, batch: OffPolicyMARLBatch):
        info_train = {}

        rewards_tot = ops.stack([r for r in batch.rewards.agent_wise.values()], axis=1).mean(dim=1)
        terminals_tot = ops.stack([d for d in batch.terminals.agent_wise.values()], axis=1).all(dim=1).float()

        ############################################
        #  Feedforward
        ############################################

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
                        q_next.grouped_tensor[batch.next_avail_actions[group] == 0] = -1e10
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

        ############################################
        #  Calculate total values and loss
        ############################################
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
        target_value = ops.stop_gradient(target_value)
        td_error = q_tot_eval - target_value

        # calculate the weights
        ones = ops.ones_like(td_error)
        w = ones * self.alpha
        if self.config.agent == "CWQMIX":
            condition_1_list = []
            for group, agent_keys in self.groups.items():
                n_agents = self.n_group_agents[group]
                mask_values = batch.valid_mask(group, n_agents).reshape([batch.batch_size, n_agents, batch.seq_length])
                a_greedy = actions_greedy.group(group)
                act = batch.actions[group].reshape([batch.batch_size, n_agents, batch.seq_length])
                condition_1_list.append(((a_greedy == act) * mask_values))
            condition_1 = ops.concat(condition_1_list, axis=1).all(dim=1).reshape(-1)
            condition_2 = target_value > q_tot_centralized
            conditions = condition_1 | condition_2
            w = ops.where(conditions, ones, w)
        elif self.config.agent == "OWQMIX":
            condition = td_error < 0
            w = ops.where(condition, ones, w)
        else:
            raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            loss_central = (((q_tot_centralized - target_value) ** 2) * filled).sum() / filled.sum()
            loss_qmix = (ops.stop_gradient(w) * (td_error ** 2) * filled).sum() / filled.sum()
        else:
            loss_central = self.mse_loss(q_tot_centralized, target_value)
            loss_qmix = (ops.stop_gradient(w) * (td_error ** 2)).mean()

        loss = loss_qmix + loss_central

        info_train.update({
            "loss_Qmix": loss_qmix.numpy(),
            "loss_central": loss_central.numpy(),
            "loss": loss.numpy(),
            "predictQ": q_tot_eval.mean().numpy()
        })

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        (loss, info_train), grads = self.grad_fn(batch)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update(info_train)
        info["learning_rate"] = lr.numpy()

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
