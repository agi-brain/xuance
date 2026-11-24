"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.mindspore import ops, Module, Tensor
from xuance.mindspore.utils import clip_grads, AgentGroupedTensor
from xuance.mindspore.learners.multi_agent_rl.isac_learner import ISAC_Learner
from xuance.mindspore.rl_models.modules import OffPolicyMARLBatch


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(MASAC_Learner, self).__init__(config, agent_grouping, model, callback)

    def forward_fn_actor(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        observations_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args
        obs_joint_t = kwargs["joint_observations_t"]

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)
        agent_keys = self.groups[group_key]

        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  group_key=group_key,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        actions_eval_agent_wise = actions_eval.agent_wise
        log_pi_eval = model_output.log_probs

        if self.use_rnn:
            actions_eval_agent_wise = {k: v[:, :-1] for k, v in actions_eval_agent_wise.items()}

        # calculate the objective of actor
        actions_all = batch.actions.agent_wise.copy()
        for key in agent_keys:
            actions_all[key] = actions_eval_agent_wise[key]
        if self.use_rnn:
            actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
        else:
            actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))

        policy_q_1, policy_q_2 = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                    joint_actions=actions_joint_eval,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group_key,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)
        log_pi_eval_i = log_pi_eval.packed(group_key).reshape(-1)
        policy_q = ops.minimum(policy_q_1.packed(group_key), policy_q_2.packed(group_key)).reshape(-1)

        loss_a = ((self.alpha[group_key] * log_pi_eval_i - policy_q) * mask_values).sum() / mask_values.sum()

        return loss_a, log_pi_eval_i, policy_q, policy_q_1, policy_q_2

    def forward_fn_critic(self, batch: OffPolicyMARLBatch, *args, **kwargs):
        agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key = args
        obs_joint = kwargs["joint_observations"]
        obs_joint_t = kwargs["joint_observations_t"]
        actions_joint_t = kwargs["joint_actions_t"]

        n_agents = self.n_group_agents[group_key]
        mask_values = batch.valid_mask(group_key, n_agents).reshape(-1)

        model_output = self.model(observations=batch.observations,
                                  agent_indices=batch.agent_indices,
                                  group_key=group_key,
                                  rnn_states=rnn_states_actor)
        actions_eval = model_output.actions
        log_pi_eval = model_output.log_probs

        action_q_1, action_q_2 = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                    joint_actions=actions_joint_t,
                                                    agent_indices=agent_indices_t,
                                                    group_key=group_key,
                                                    rnn_states_1=rnn_states_critic_1,
                                                    rnn_states_2=rnn_states_critic_2)

        if self.use_rnn:
            actions_joint_next = self.get_joint_input(actions_eval.agent_wise,
                                                      output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        group_key=group_key,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            log_pi_next = AgentGroupedTensor(
                {k: v[:, :, 1:] for k, v in log_pi_eval.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            next_model_output = self.model(observations=batch.next_observations,
                                           agent_indices=batch.agent_indices)
            actions_next = next_model_output.actions
            log_pi_next = next_model_output.log_probs
            actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
            obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                  output_shape=(batch.batch_size, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices)

        action_q_1_i = action_q_1.packed(group_key).reshape(-1)
        action_q_2_i = action_q_2.packed(group_key).reshape(-1)
        log_pi_next_eval = log_pi_next.packed(group_key).reshape(-1)
        rewards = batch.rewards.packed(group_key).reshape(-1)
        terminals = batch.terminals.packed(group_key).reshape(-1)

        target_value = q_next.packed(group_key).reshape(-1) - self.alpha[group_key] * log_pi_next_eval
        backup = rewards + (1 - terminals) * self.gamma * target_value
        backup = ops.stop_gradient(backup)
        td_error_1 = action_q_1_i - backup
        td_error_1 *= mask_values
        td_error_2 = action_q_2_i - backup
        td_error_2 *= mask_values

        loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
        return loss_c, action_q_1_i, action_q_2_i, td_error_1, td_error_2

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        if self.use_rnn:
            obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                             output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            obs_joint_t = obs_joint[:, :-1]
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise,
                                                   output_shape=(batch.batch_size, batch.seq_length, -1))
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            obs_joint_t = obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                                           output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise, output_shape=(batch.batch_size, -1))
            agent_indices_t = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        for group_key in self.group_keys:
            # update critic
            (loss_c, action_q_1_i, action_q_2_i, td_error_1, td_error_2), grads_critic = self.grad_fn_critic[group_key](
                batch, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2, group_key,
                joint_observations=obs_joint, joint_observations_t=obs_joint_t, joint_actions_t=actions_joint_t
            )
            if self.use_grad_clip:
                grads_critic = clip_grads(grads_critic, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['critic'](grads_critic)

            # update actor
            (loss_a, log_pi_eval_i, policy_q, policy_q_1, policy_q_2), grads_actor = self.grad_fn_actor[group_key](
                batch, obs_joint_t, agent_indices_t, rnn_states_actor, rnn_states_critic_1, rnn_states_critic_2,
                group_key, joint_observations_t=obs_joint_t)
            if self.use_grad_clip:
                grads_actor = clip_grads(grads_actor, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))
            self.optimizer[group_key]['actor'](grads_actor)

            # automatic entropy tuning
            if self.use_automatic_entropy_tuning:
                (alpha_loss, _), grads_alpha = self.grad_fn_alpha[group_key](log_pi_eval_i, group_key)
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
                                                           policy_q_1=policy_q_1, policy_q_2=policy_q_2,
                                                           log_pi_eval_i=log_pi_eval_i, policy_q=policy_q))

        self.model.soft_update(self.tau)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))
        return info
