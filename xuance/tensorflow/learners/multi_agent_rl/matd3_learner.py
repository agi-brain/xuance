"""
Multi-Agent TD3

"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.learners.multi_agent_rl.itd3_learner import ITD3_Learner
from xuance.tensorflow.rl_models.modules import OffPolicyMARLBatch


class MATD3_Learner(ITD3_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        ITD3_Learner.__init__(self, config, agent_grouping, model, callback)

    @tf.function
    def forward_fn(self, **kwargs):
        info_train = {}

        #########################################
        # Prepare training data
        #########################################

        batch = OffPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
            observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
            actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
            rewards=AgentGroupedTensor(kwargs["rewards"], self.agent_grouping),
            terminals=AgentGroupedTensor(kwargs["terminals"], self.agent_grouping),
            agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
            agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
        )
        if self.use_actions_mask:
            batch.avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
            if not self.use_rnn:
                batch.next_avail_actions = AgentGroupedTensor(kwargs["next_avail_actions"], self.agent_grouping)
        if self.use_rnn:
            batch.filled_masks = kwargs["filled_masks"]
        else:
            batch.next_observations = AgentGroupedTensor(kwargs["next_observations"], self.agent_grouping)

        if self.use_rnn:
            obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                             output_shape=(batch.batch_size, batch.seq_length + 1, -1))
            obs_joint_t = obs_joint[:, :-1]
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise,
                                                   output_shape=(batch.batch_size, batch.seq_length, -1))
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            obs_joint_t = obs_joint = self.get_joint_input(batch.observations.agent_wise,
                                                           output_shape=(batch.batch_size, -1))
            actions_joint_t = self.get_joint_input(batch.actions.agent_wise, output_shape=(batch.batch_size, -1))
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        # initial hidden states for rnn
        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        if self.use_rnn:
            actions_next = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              rnn_states=rnn_states_actor)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise,
                                                      (batch.batch_size, batch.seq_length + 1, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            actions_next = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices)
            actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
            obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                  output_shape=(batch.batch_size, -1))
            q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                        joint_actions=actions_joint_next,
                                        agent_indices=batch.agent_indices)

        #########################################
        # Feedforward & backpropogation
        #########################################

        # update critic
        with tf.GradientTape(persistent=True) as tape_critic:

            q_eval_A, q_eval_B, _ = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                       joint_actions=actions_joint_t,
                                                       agent_indices=agent_indices_t,
                                                       rnn_states_1=rnn_states_critic_1,
                                                       rnn_states_2=rnn_states_critic_2)

            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                q_eval_A_i = tf.reshape(q_eval_A.packed(group), [-1])
                q_eval_B_i = tf.reshape(q_eval_B.packed(group), [-1])
                q_next_i = tf.reshape(q_next.packed(group), [-1])
                rewards_i = tf.reshape(batch.rewards.packed(group), [-1])
                terminals_i = tf.reshape(batch.terminals.packed(group), [-1])
                q_target = rewards_i + (1 - terminals_i) * self.gamma * q_next_i
                q_target = tf.stop_gradient(q_target)

                td_error_A = (q_eval_A_i - q_target) * mask_values
                td_error_B = (q_eval_B_i - q_target) * mask_values

                loss_c = (tf.reduce_sum(td_error_A ** 2) + tf.reduce_sum(td_error_B ** 2)) / tf.reduce_sum(
                    mask_values)

                gradients = tape_critic.gradient(loss_c, self.model.critics[group].trainable_variables)
                if self.use_grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.optimizer[group]['critic'].apply_gradients(
                        zip(gradients, self.model.critics[group].trainable_variables))
                else:
                    self.optimizer[group]['critic'].apply_gradients(
                        zip(gradients, self.model.critics[group].trainable_variables))

        # update actor(s)
        if self.iterations % self.actor_update_delay == 0:

            with tf.GradientTape(persistent=True) as tape_actor:

                actions_eval = self.model(observations=observations_t,
                                          agent_indices=agent_indices_t,
                                          rnn_states=rnn_states_actor).actions
                actions_eval_agent_wise = actions_eval.agent_wise
                for group, n_agents in self.n_group_agents.items():
                    mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])
                    agent_keys = self.groups[group]

                    # calculate the objective of actor
                    actions_all = batch.actions.agent_wise.copy()
                    for key in agent_keys:
                        actions_all[key] = actions_eval_agent_wise[key]
                    if self.use_rnn:
                        actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
                    else:
                        actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
                    _, _, q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                        joint_actions=actions_joint_eval,
                                                        agent_indices=agent_indices_t,
                                                        group_key=group,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)
                    q_policy_i = tf.reshape(q_policy.packed(group), [-1])

                    loss_actor = -tf.reduce_sum(q_policy_i * mask_values) / tf.reduce_sum(mask_values)

                    gradients = tape_actor.gradient(loss_actor, self.model.actors[group].trainable_variables)
                    if self.use_grad_clip:
                        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                        self.optimizer['actor'].apply_gradients(
                            zip(gradients, self.model.actors[group].trainable_variables))
                    else:
                        self.optimizer['actor'].apply_gradients(
                            zip(gradients, self.model.actors[group].trainable_variables))

            self.model.soft_update(self.tau)

        return info_train

