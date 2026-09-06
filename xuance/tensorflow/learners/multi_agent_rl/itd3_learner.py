"""
Independent TD3 for multi-agent cooperative task
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.learners import OffPolicyMultiAgentLearner
from xuance.tensorflow.rl_models.modules import OffPolicyMARLBatch


class ITD3_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(ITD3_Learner, self).__init__(config, agent_grouping, model, callback)
        self.tau = config.tau
        self.actor_update_delay = config.actor_update_delay

    def build_optimizer(self):
        self.optimizer = {
            key: {'actor': keras.optimizers.Adam(self.config.learning_rate_actor),
                  'critic': keras.optimizers.Adam(self.config.learning_rate_critic)}
            for key in self.group_keys
        }

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

        rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
        rnn_states_critic_1, rnn_states_critic_2 = self.model.init_critic_rnn_states(batch.batch_size)

        if self.use_rnn:
            observations_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.observations.grouped_tensor.items()}, self.agent_grouping
            )
            agent_indices_t = AgentGroupedTensor(
                {k: v[:, :, :-1] for k, v in batch.agent_indices.grouped_tensor.items()}, self.agent_grouping
            )
        else:
            observations_t = batch.observations
            agent_indices_t = batch.agent_indices

        if self.use_rnn:
            next_actions = self.model.Atarget(observations=batch.observations,
                                              agent_indices=batch.agent_indices,
                                              rnn_states=rnn_states_actor)
            q_next = self.model.Qtarget(observations=batch.observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices,
                                        rnn_states_1=rnn_states_critic_1,
                                        rnn_states_2=rnn_states_critic_2)
            q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
        else:
            next_actions = self.model.Atarget(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices)
            q_next = self.model.Qtarget(observations=batch.next_observations,
                                        actions=next_actions,
                                        agent_indices=batch.agent_indices)

        #########################################
        # Feedforward & backpropogation
        #########################################

        # update critic
        with tf.GradientTape(persistent=True) as tape_critic:

            q_eval_A, q_eval_B, _ = self.model.Qpolicy(observations=observations_t,
                                                       actions=batch.actions,
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

                loss_c = (tf.reduce_sum(td_error_A ** 2) + tf.reduce_sum(td_error_B ** 2)) / tf.reduce_sum(mask_values)

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

                for group, n_agents in self.n_group_agents.items():
                    mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                    _, _, q_policy = self.model.Qpolicy(observations=observations_t,
                                                        actions=actions_eval,
                                                        agent_indices=agent_indices_t,
                                                        grou_key=group,
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

    @tf.function
    def learn(self, **kwargs):
        if self.distributed_training:
            info_train = self.model.mirrored_strategy.run(self.forward_fn, kwargs=kwargs)
            return info_train[0]
        else:
            return self.forward_fn(**kwargs)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        inputs_learn = {
            "batch_size": batch.batch_size,
            "seq_length": batch.seq_length,
            "observations": batch.observations.grouped_tensor,
            "actions": batch.actions.grouped_tensor,
            "rewards": batch.rewards.grouped_tensor,
            "terminals": batch.terminals.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor
            if not self.use_rnn:
                inputs_learn["next_avail_actions"] = batch.next_avail_actions.grouped_tensor

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks
        else:
            inputs_learn["next_observations"] = batch.next_observations.grouped_tensor

        info_train = self.learn(**inputs_learn)

        info.update(info_train)
        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
