"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: TensorFlow 2.X
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
from xuance.tensorflow import tf, keras
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.learners.multi_agent_rl.iddpg_learner import IDDPG_Learner
from xuance.tensorflow.rl_models.modules import OffPolicyMARLBatch


class MADDPG_Learner(IDDPG_Learner):

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
        rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

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

        #########################################
        # Feedforward & backpropogation
        #########################################
        gradients_a, gradients_c = {}, {}
        with tf.GradientTape(persistent=True) as tape:

            # feedforward
            actions_eval = self.model(observations=observations_t,
                                      agent_indices=agent_indices_t,
                                      rnn_states=rnn_states_actor).actions
            actions_eval_agent_wise = actions_eval.agent_wise
            q_eval = self.model.Qpolicy(joint_observations=obs_joint_t,
                                        joint_actions=actions_joint_t,
                                        agent_indices=agent_indices_t,
                                        rnn_states=rnn_states_critic)

            if self.use_rnn:
                actions_next = self.model.Atarget(observations=batch.observations,
                                                  agent_indices=batch.agent_indices,
                                                  rnn_states=rnn_states_actor)
                actions_joint_next = self.get_joint_input(actions_next.agent_wise,
                                                          (batch.batch_size, batch.seq_length + 1, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices,
                                            rnn_states=rnn_states_critic)
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
            else:
                actions_next = self.model.Atarget(observations=batch.next_observations,
                                                  agent_indices=batch.agent_indices)
                actions_joint_next = self.get_joint_input(actions_next.agent_wise, (batch.batch_size, -1))
                obs_joint_next = self.get_joint_input(batch.next_observations.agent_wise,
                                                      output_shape=(batch.batch_size, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint_next,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices,
                                            rnn_states=rnn_states_critic)

            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])
                agent_keys = self.groups[group]

                # update critic
                q_eval_a = tf.reshape(q_eval.packed(group), [-1])
                q_next_i = tf.reshape(q_next.packed(group), [-1])
                rewards = tf.reshape(batch.rewards.packed(group), [-1])
                terminals = tf.reshape(batch.terminals.packed(group), [-1])

                q_target = rewards + (1 - terminals) * self.gamma * q_next_i
                q_target = tf.stop_gradient(q_target)
                td_error = (q_eval_a - q_target) * mask_values

                loss_critic = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(mask_values)

                gradients_c[group] = tape.gradient(loss_critic, self.model.critics[group].trainable_variables)
                if self.use_grad_clip:
                    gradients_c[group], _ = tf.clip_by_global_norm(gradients_c[group], clip_norm=self.grad_clip_norm)
                    self.optimizer[group]['critic'].apply_gradients(zip(gradients_c[group],
                                                                        self.model.critics[group].trainable_variables))
                else:
                    self.optimizer[group]['critic'].apply_gradients(zip(gradients_c[group],
                                                                        self.model.critics[group].trainable_variables))

                # update actor
                # calculate the objective of actor
                actions_all = batch.actions.agent_wise.copy()
                for key in agent_keys:
                    actions_all[key] = actions_eval_agent_wise[key]
                if self.use_rnn:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, batch.seq_length, -1))
                else:
                    actions_joint_eval = self.get_joint_input(actions_all, (batch.batch_size, -1))
                q_policy = self.model.Qpolicy(joint_observations=obs_joint_t,
                                              joint_actions=actions_joint_eval,
                                              agent_indices=agent_indices_t,
                                              group_key=group,
                                              rnn_states=rnn_states_critic)
                q_policy_i = tf.reshape(q_policy.packed(group), [-1])
                loss_actor = -tf.reduce_sum(q_policy_i * mask_values) / tf.reduce_sum(mask_values)

                gradients_a[group] = tape.gradient(loss_actor, self.model.actors[group].trainable_variables)
                if self.use_grad_clip:
                    gradients_a[group], _ = tf.clip_by_global_norm(gradients_a[group], clip_norm=self.grad_clip_norm)
                    self.optimizer[group]['actor'].apply_gradients(zip(gradients_a[group],
                                                                       self.model.actors[group].trainable_variables))
                else:
                    self.optimizer[group]['actor'].apply_gradients(zip(gradients_a[group],
                                                                       self.model.actors[group].trainable_variables))

                info_train.update({
                    f"{group}/loss_actor": loss_actor,
                    f"{group}/loss_critic": loss_critic,
                    f"{group}/predictQ": tf.reduce_mean(q_eval_a)
                })

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
        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.model, info=info))

        return info
