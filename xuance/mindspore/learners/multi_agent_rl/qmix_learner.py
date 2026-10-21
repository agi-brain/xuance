"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link:
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, Module, Tensor, ops
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class QMIX_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(QMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()

    def forward_fn(self, batch: OffPolicyMARLBatch):
        info_train = {}

        rewards_tot = ops.stack([r for r in batch.rewards.agent_wise.values()], axis=1).mean(dim=1)
        terminals_tot = ops.stack([d for d in batch.terminals.agent_wise.values()], axis=1).all(dim=1).float()

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
            mask_values = batch.valid_mask(group, n_agents).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

            actions_taken = batch.actions.group(group)
            q_eval_taken = q_eval.group(group).gather(-1, actions_taken.long().unsqueeze(-1)).reshape(
                [batch.batch_size, n_agents, batch.seq_length])

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

        if self.use_rnn:
            q_tot_eval = self.model.Q_tot(q_eval_a, batch.global_states[:, :-1]).reshape(-1)
            q_tot_next = self.model.Qtarget_tot(q_next_a, batch.global_states[:, 1:]).reshape(-1)
        else:
            q_tot_eval = self.model.Q_tot(q_eval_a, batch.global_states).reshape(-1)
            q_tot_next = self.model.Qtarget_tot(q_next_a, batch.next_global_states).reshape(-1)

        rewards_tot = rewards_tot.reshape(-1)
        terminals_tot = terminals_tot.reshape(-1)

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next
        q_tot_target = ops.stop_gradient(q_tot_target)

        # calculate the loss
        if self.use_rnn:
            filled = batch.filled_masks.reshape(-1)
            td_errors = (q_tot_eval - q_tot_target) * filled
            loss = (td_errors ** 2).sum() / filled.sum()
        else:
            loss = self.mse_loss(q_tot_eval, q_tot_target)

        info_train.update({
            "loss_Q": loss.numpy(),
            "predictQ": q_tot_eval.mean().numpy()
        })

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True)

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
