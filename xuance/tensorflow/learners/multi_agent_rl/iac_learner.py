"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow2
"""
from xuance.tensorflow import tf, keras
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OnPolicyMARLBatch
from xuance.tensorflow.learners import OnPolicyMultiAgentLearner


class IAC_Learner(OnPolicyMultiAgentLearner):

    @tf.function
    def forward_fn(self, **kwargs):

        info_train, gradients = {}, {}

        batch = OnPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
            observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
            actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
            values=AgentGroupedTensor(kwargs["values"], self.agent_grouping),
            returns=AgentGroupedTensor(kwargs["returns"], self.agent_grouping),
            advantages=AgentGroupedTensor(kwargs["advantages"], self.agent_grouping),
            agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
            agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
        )
        if self.use_actions_mask:
            batch.avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)

        if self.use_rnn:
            batch.filled_masks = kwargs["filled_masks"]

        with tf.GradientTape() as tape:
            # initial hidden states for rnn
            rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
            rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

            # feedforward
            policy_outputs = self.model(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                avail_actions=batch.avail_actions,
                rnn_states=rnn_states_actor
            )
            value_outputs = self.model.get_values(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                rnn_states=rnn_states_critic
            )
            values_pred = value_outputs.values

            # calculate losses and update networks for each group of agents
            loss_individual = []
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                # actor loss
                dist = policy_outputs.distributions[group]
                log_pi = tf.reshape(dist.log_prob(batch.actions.packed(group)), [-1])
                advantages = tf.reshape(batch.advantages.group(group), [-1])
                advantages = tf.stop_gradient(advantages)

                masked_actor_loss = (advantages * log_pi) * mask_values
                actor_loss = -tf.reduce_sum(masked_actor_loss) / tf.reduce_sum(mask_values)

                # entropy loss
                entropy = tf.reshape(dist.entropy(), [-1])
                entropy_loss = tf.reduce_sum(entropy * mask_values) / tf.reduce_sum(mask_values)

                # value loss
                value_pred_i = tf.reshape(values_pred.packed(group), [-1])
                value_target = tf.reshape(batch.returns.packed(group), [-1])
                values_i = tf.reshape(batch.values.packed(group), [-1])
                if self.use_value_clip:
                    value_clipped = values_i + tf.clip_by_value(
                        value_pred_i - values_i,
                        clip_value_min=-self.value_clip_range,
                        clip_value_max=self.value_clip_range
                    )
                    if self.use_value_norm:
                        self.value_normalizer[group].update(tf.reshape(value_target, [-1, 1]))
                        value_target = self.value_normalizer[group].normalize(tf.reshape(value_target, [-1, 1]))
                        value_target = tf.reshape(value_target, [-1])
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred_i, value_target)
                        loss_v_clipped = self.huber_loss(value_clipped, value_target)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c_ = tf.maximum(loss_v, loss_v_clipped)
                    critic_loss = tf.reduce_sum(loss_c_ * mask_values) / tf.reduce_sum(mask_values)
                else:
                    if self.use_value_norm:
                        self.value_normalizer[group].update(value_target)
                        value_target = self.value_normalizer[group].normalize(value_target)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred_i, value_target)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                    critic_loss = tf.reduce_sum(loss_v * mask_values) / tf.reduce_sum(mask_values)

                loss_i = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
                loss_individual.append(loss_i)

                info_train.update({
                    f"{group}/actor_loss": actor_loss,
                    f"{group}/critic_loss": critic_loss,
                    f"{group}/entropy": entropy_loss,
                    f"{group}/individual_loss": loss_i,
                    f"{group}/predict_value": tf.reduce_mean(value_pred_i),
                })

            loss = sum(loss_individual)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            info_train.update({f"{group}/total_loss": loss})

        return info_train

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
            "values": batch.values.grouped_tensor,
            "returns": batch.returns.grouped_tensor,
            "advantages": batch.advantages.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks

        info_train = self.learn(**inputs_learn)

        info.update(info_train)

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
