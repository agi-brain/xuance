"""
Independent Q-learning (IQL)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, ops, Module, Tensor
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch
from xuance.mindspore.learners import OffPolicyMultiAgentLearner
from xuance.mindspore.utils import clip_grads


class IQL_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(IQL_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()

    def forward_fn(self, batch: OffPolicyMARLBatch):
        info_train = {}

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

        # calculate losses and update networks for each group of agents
        loss_group = []
        for group, n_agents in self.n_group_agents.items():
            mask_values = batch.valid_mask(group, n_agents).reshape(-1)

            rewards = batch.rewards.packed(group).reshape(-1)
            terminals = batch.terminals.packed(group).reshape(-1)

            actions_taken = batch.actions.packed(group)
            q_eval_taken = q_eval.packed(group).gather(-1, actions_taken.long().unsqueeze(-1)).reshape(-1)

            if self.use_actions_mask:
                if self.use_rnn:
                    next_avail_actions = batch.avail_actions.group(group)[:, 1:]
                else:
                    next_avail_actions = batch.next_avail_actions.group(group)
                q_next.group(group)[next_avail_actions == 0] = -1e10

            if self.config.double_q:
                actions_next_taken = actions_next.packed(group)
                q_next_taken = q_next.packed(group).gather(-1, actions_next_taken.long().unsqueeze(-1)).reshape(-1)
            else:
                q_next_taken = q_next.packed(group).max(dim=-1, keepdim=True).values.reshape(-1)

            q_target = rewards + (1 - terminals) * self.gamma * q_next_taken
            q_target = ops.stop_gradient(q_target)

            # calculate the loss function
            td_error = (q_eval_taken - q_target) * mask_values
            masked_mean_loss_group = (td_error ** 2).sum() / mask_values.sum()
            loss_group.append(masked_mean_loss_group)

            info_train.update({
                f"{group}/loss_Q": masked_mean_loss_group.numpy(),
                f"{group}/predictQ": q_eval_taken.mean().numpy()
            })

        loss = sum(loss_group)
        info_train.update({"loss": loss.numpy()})

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask)

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
