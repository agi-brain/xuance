"""
Independent Soft Actor-critic (ISAC) with discrete action spaces.
Implementation: Pytorch
"""
import math
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ops, Module, Tensor
from xuance.mindspore.utils import clip_grads, AgentGroupedTensor
from xuance.mindspore.learners.multi_agent_rl.isac_learner import ISAC_Learner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class ISACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(ISACDIS_Learner, self).__init__(config, agent_grouping, model, callback)

    def get_target_entropy(self):
        ratio = getattr(self.config, "target_entropy_ratio", 0.8)
        return {
            key: ratio * math.log(self.model.actors[key].action_dim)
            for key in self.group_keys
        }

    def forward_fn_alpha(self, *args):
        batch, action_probs, log_action_probs, group_key = args
        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        entropy = -(action_probs * log_action_probs).sum(dim=-1).reshape(-1)
        entropy_error = entropy - self.target_entropy[group_key]
        entropy_error = ops.stop_gradient(entropy_error)
        alpha_loss = (self.log_alpha[group_key] * entropy_error * mask_values).sum() / mask_values.sum()
        return alpha_loss, self.log_alpha[group_key]

    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, avail_actions_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        action_probs, log_action_probs, policy_q = self.model.Qpolicy(
            observations=observations_t,
            agent_indices=agent_indices_t,
            avail_actions=avail_actions_t,
            group_key=group_key,
            rnn_states_actor=rnn_states_actor,
            rnn_states_critic_1=rnn_states_critic_1,
            rnn_states_critic_2=rnn_states_critic_2
        )
        action_probs_i = action_probs.group(group_key)
        log_action_probs_i = log_action_probs.group(group_key)
        policy_q_i = policy_q.group(group_key)

        loss_a = self.alpha[group_key] * (action_probs_i * log_action_probs_i).sum(dim=-1)
        loss_a = loss_a - policy_q_i
        loss_a = (loss_a * mask_values).sum() / mask_values.sum()
        return loss_a, action_probs_i, log_action_probs_i, policy_q_i

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args

        action_q_1, action_q_2 = self.model.Qvalues(observations=observations_t,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group_key,
                                                    rnn_states_critic_1=rnn_states_critic_1,
                                                    rnn_states_critic_2=rnn_states_critic_2)

        if self.use_rnn:
            action_prob_next, log_pi_next, q_next = self.model.Qtarget(next_observations=batch.observations,
                                                                       agent_indices=batch.agent_indices,
                                                                       group_key=group_key,
                                                                       rnn_states_critic_1=rnn_states_critic_1,
                                                                       rnn_states_critic_2=rnn_states_critic_2)
            action_prob_next.grouped_tensor = {k: v[:, :, 1:] for k, v in action_prob_next.grouped_tensor.items()}
            log_pi_next.grouped_tensor = {k: v[:, :, 1:] for k, v in log_pi_next.grouped_tensor.items()}
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            action_prob_next, log_pi_next, q_next = self.model.Qtarget(next_observations=batch.next_observations,
                                                                       agent_indices=batch.agent_indices,
                                                                       group_key=group_key)

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        # update critic
        action_q_1_a = action_q_1.packed(group_key).gather(-1, batch.actions.packed(group_key).long().unsqueeze(-1))
        action_q_1_a = action_q_1_a.reshape(-1)
        action_q_2_a = action_q_2.packed(group_key).gather(-1, batch.actions.packed(group_key).long().unsqueeze(-1))
        action_q_2_a = action_q_2_a.reshape(-1)
        target_value = (action_prob_next.group(group_key) * self.alpha[group_key] * log_pi_next.group(group_key))
        target_value = q_next.group(group_key) - target_value.sum(dim=-1)
        target_value = target_value.reshape(-1)
        backup = batch.rewards.packed(group_key) + (1 - batch.terminals.packed(group_key)) * self.gamma * target_value
        backup = ops.stop_gradient(backup)
        td_error_1 = action_q_1_a - backup
        td_error_1 *= mask_values
        td_error_2 = action_q_2_a - backup
        td_error_2 *= mask_values
        loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()

        return loss_c, action_q_1_a, action_q_2_a, td_error_1, td_error_2

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        if self.use_rnn:
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
            avail_actions_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.avail_actions.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices
            avail_actions_t = batch.avail_actions

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        for group_key in self.group_keys:
            # update critic
            (loss_c, action_q_1_i, action_q_2_i, td_error_1, td_error_2), grads_critic = self.grad_fn_critic[group_key](
                batch, observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2,
                group_key)
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            # update actor
            (loss_a, action_probs, log_action_probs, policy_q), grads_actor = self.grad_fn_actor[group_key](
                batch, observations_t, avail_actions_t, agent_indices_t,
                rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['actor'](grads_actor)

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                (alpha_loss, _), grads_alpha = self.grad_fn_alpha[group_key](
                    batch, action_probs, log_action_probs, group_key)
                self.alpha_optimizer[group_key](grads_alpha)
                self.alpha[group_key] = ops.exp(self.log_alpha[group_key])
            else:
                alpha_loss = 0

            learning_rate_actor = self.scheduler[group_key]['actor'].get_last_lr()[0]
            learning_rate_critic = self.scheduler[group_key]['critic'].get_last_lr()[0]

            info.update({
                f"{group_key}/learning_rate_actor": learning_rate_actor.asnumpy(),
                f"{group_key}/learning_rate_critic": learning_rate_critic.asnumpy(),
                f"{group_key}/loss_actor": loss_a.asnumpy(),
                f"{group_key}/loss_critic": loss_c.asnumpy(),
                f"{group_key}/predictQ": policy_q.mean().asnumpy(),
                f"{group_key}/alpha_loss": alpha_loss.asnumpy(),
                f"{group_key}/alpha": self.alpha[group_key].asnumpy(),
            })

            info.update(self.callback.on_update_agent_wise(self.iterations, group_key, info=info,
                                                           action_q_1_i=action_q_1_i, action_q_2_i=action_q_2_i,
                                                           td_error_1=td_error_1, td_error_2=td_error_2,
                                                           policy_q=policy_q,
                                                           action_probs=action_probs,
                                                           log_action_probs=log_action_probs))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
