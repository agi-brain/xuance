"""
Independent Q-learning (IQL)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.rl_models.modules import AgentGroupedTensor, OffPolicyMARLBatch
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

    def build_optimizer(self):
        self.optimizer = optim.Adam(params=self.model.individual_q_networks.trainable_params(),
                                    lr=self.config.learning_rate,
                                    eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                     start_factor=1.0,
                                                     end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)

    def forward_fn(self, batch: OffPolicyMARLBatch):
        info_train = {}

        # batch = OffPolicyMARLBatch(
        #     batch_size=kwargs["batch_size"],
        #     seq_length=kwargs["seq_length"],
        #     observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
        #     actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
        #     rewards=AgentGroupedTensor(kwargs["rewards"], self.agent_grouping),
        #     terminals=AgentGroupedTensor(kwargs["terminals"], self.agent_grouping),
        #     agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
        #     agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
        # )
        # if self.use_actions_mask:
        #     batch.avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
        #     if not self.use_rnn:
        #         batch.next_avail_actions = AgentGroupedTensor(kwargs["next_avail_actions"], self.agent_grouping)
        # if self.use_rnn:
        #     batch.filled_masks = kwargs["filled_masks"]
        # else:
        #     batch.next_observations = AgentGroupedTensor(kwargs["next_observations"], self.agent_grouping)

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
                f"{group}/loss_Q": masked_mean_loss_group,
                f"{group}/predictQ": q_eval_taken.mean()
            })

        loss = sum(loss_group)

        return loss, info_train

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        (loss, inf_train), grads = self.grad_fn(batch)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        # # initialize rnn hidden states when use rnn
        # rnn_states = self.model.init_rnn_states(batch.batch_size)
        #
        # _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)
        #
        # for key in self.model_keys:
        #     mask_values = agent_mask[key]
        #
        #     if self.use_actions_mask:
        #         q_next[key][avail_actions_next[key] == 0] = -1e10
        #
        #     if self.config.double_q:
        #         _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
        #         q_next_a = q_next[key].gather(actions_next_greedy[key].unsqueeze(-1).long(), -1, -1).reshape(bs)
        #     else:
        #         q_next_a = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)
        #
        #     q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a
        #
        #     (loss, q_eval_a, td_error), grads = self.grad_fn[key](obs, actions, mask_values,
        #                                                           avail_actions, IDs, q_target, key)
        #     if self.use_grad_clip:
        #         grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
        #     self.optimizer[key](grads)
        #
        #     self.scheduler[key].step()
        #     lr = self.scheduler[key].get_last_lr()[0]
        #
        #     info.update({
        #         f"{key}/learning_rate": lr.asnumpy(),
        #         f"{key}/loss_Q": loss.asnumpy(),
        #         f"{key}/predictQ": q_eval_a.mean().asnumpy()
        #     })
        #
        #     info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update",
        #                                                    mask_values=mask_values, q_eval_a=q_eval_a,
        #                                                    q_next_a=q_next_a, q_target=q_target,
        #                                                    td_error=td_error, loss=loss))

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", model=self.model, info=info))

        return info
