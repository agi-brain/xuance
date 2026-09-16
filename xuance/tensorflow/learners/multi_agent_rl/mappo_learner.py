"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OnPolicyMARLBatch
from xuance.tensorflow.learners import OnPolicyMultiAgentLearner


class MAPPO_Learner(OnPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(MAPPO_Learner, self).__init__(config, agent_grouping, model, callback)
        self.clip_range = config.clip_range
        self.use_global_state = config.use_global_state

    @tf.function
    def forward_fn(self, **kwargs):

        info_train, gradients = {}, {}

        batch = OnPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
            global_states=kwargs["global_states"],
            observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
            actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
            values=AgentGroupedTensor(kwargs["values"], self.agent_grouping),
            returns=AgentGroupedTensor(kwargs["returns"], self.agent_grouping),
            advantages=AgentGroupedTensor(kwargs["advantages"], self.agent_grouping),
            agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
            agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
            old_log_probs=AgentGroupedTensor(kwargs["old_log_probs"], self.agent_grouping)
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
                states=batch.global_states,
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                rnn_states=rnn_states_critic
            )
            values_pred = value_outputs.values

            # calculate losses for each agent
            actor_loss, entropy_loss, critic_loss = [], [], []
            running_mean_dict, running_mean_sq_dict, debiasing_term_dict = {}, {}, {}
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                # actor loss
                dist = policy_outputs.distributions[group]
                log_pi = tf.reshape(dist.log_prob(batch.actions.packed(group)), [-1])
                advantages = tf.reshape(batch.advantages.group(group), [-1])
                advantages = tf.stop_gradient(advantages)
                old_log_prob = tf.reshape(batch.old_log_probs.packed(group), [-1])

                ratio = tf.exp(log_pi - old_log_prob)
                surrogate1 = ratio * advantages
                surrogate2 = tf.clip_by_value(
                    ratio,
                    clip_value_min=1.0 - self.clip_range,
                    clip_value_max=1.0 + self.clip_range
                ) * advantages

                masked_actor_loss = tf.reduce_sum(tf.minimum(surrogate1, surrogate2) * mask_values)
                actor_loss.append(-masked_actor_loss / tf.reduce_sum(mask_values))

                # entropy loss
                entropy = tf.reshape(dist.entropy(), [-1])
                entropy_loss.append(tf.reduce_sum(entropy * mask_values) / tf.reduce_sum(mask_values))

                # critic loss
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
                        running_mean, running_mean_sq, debiasing_term = self.value_normalizer[group].update(
                            tf.reshape(value_target, [-1, 1]),
                            running_mean=kwargs["running_mean"][group],
                            running_mean_sq=kwargs["running_mean_sq"][group],
                            debiasing_term=kwargs["debiasing_term"][group]
                        )
                        running_mean_dict[group] = running_mean
                        running_mean_sq_dict[group] = running_mean_sq
                        debiasing_term_dict[group] = debiasing_term
                        value_target = self.value_normalizer[group].normalize(
                            tf.reshape(value_target, [-1, 1]),
                            running_mean=running_mean,
                            running_mean_sq=running_mean_sq,
                            debiasing_term=debiasing_term
                        )
                        value_target = tf.reshape(value_target, [-1])
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred_i, value_target)
                        loss_v_clipped = self.huber_loss(value_clipped, value_target)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c_ = tf.maximum(loss_v, loss_v_clipped)
                    loss_c = tf.reduce_sum(loss_c_ * mask_values) / tf.reduce_sum(mask_values)
                else:
                    if self.use_value_norm:
                        running_mean, running_mean_sq, debiasing_term = self.value_normalizer[group].update(
                            value_target,
                            running_mean=kwargs["running_mean"][group],
                            running_mean_sq=kwargs["running_mean_sq"][group],
                            debiasing_term=kwargs["debiasing_term"][group]
                        )
                        running_mean_dict[group] = running_mean
                        running_mean_sq_dict[group] = running_mean_sq
                        debiasing_term_dict[group] = debiasing_term
                        value_target = self.value_normalizer[group].normalize(
                            value_target,
                            running_mean=running_mean,
                            running_mean_sq=running_mean_sq,
                            debiasing_term=debiasing_term
                        )
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred_i, value_target)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                    loss_c = tf.reduce_sum(loss_v * mask_values) / tf.reduce_sum(mask_values)

                critic_loss.append(loss_c)

                info_train.update({
                    f"{group}/actor_loss": actor_loss[-1],
                    f"{group}/critic_loss": critic_loss[-1],
                    f"{group}/entropy": entropy_loss[-1],
                    f"{group}/predict_value": tf.reduce_mean(value_pred_i)
                })

            loss = sum(actor_loss) + self.vf_coef * sum(critic_loss) - self.ent_coef * sum(entropy_loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            info_train.update({
                "total_loss": loss,
                "running_mean": running_mean_dict,
                "running_mean_sq": running_mean_sq_dict,
                "debiasing_term": debiasing_term_dict
            })

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
            "global_states": batch.global_states,
            "observations": batch.observations.grouped_tensor,
            "actions": batch.actions.grouped_tensor,
            "values": batch.values.grouped_tensor,
            "returns": batch.returns.grouped_tensor,
            "advantages": batch.advantages.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
            "old_log_probs": batch.old_log_probs.grouped_tensor
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks

        if self.use_value_norm:
            inputs_learn["running_mean"] = {
                k: tf.cast(v.running_mean, dtype=tf.float32) for k, v in self.value_normalizer.items()
            }
            inputs_learn["running_mean_sq"] = {
                k: tf.cast(v.running_mean_sq, dtype=tf.float32) for k, v in self.value_normalizer.items()
            }
            inputs_learn["debiasing_term"] = {
                k: tf.cast(v.debiasing_term, dtype=tf.float32) for k, v in self.value_normalizer.items()
            }

        info_train = self.learn(**inputs_learn)

        if self.use_value_norm:
            for group in self.group_keys:
                self.value_normalizer[group].update_tensor(
                    info_train["running_mean"][group],
                    info_train["running_mean_sq"][group],
                    info_train["debiasing_term"][group]
                )

        del info_train["running_mean"]
        del info_train["running_mean_sq"]
        del info_train["debiasing_term"]

        info.update(info_train)

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
