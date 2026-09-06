"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.utils import AgentGroupedTensor
from xuance.tensorflow.learners.multi_agent_rl.isac_learner import ISAC_Learner
from xuance.tensorflow.rl_models.modules import OffPolicyMARLBatch


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(MASAC_Learner, self).__init__(config, agent_grouping, model, callback)

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

        #########################################
        # Feedforward & backpropogation
        #########################################

        with tf.GradientTape(persistent=True) as tape:

            # feedforward
            model_output = self.model(observations=batch.observations,
                                      agent_indices=batch.agent_indices,
                                      rnn_states=rnn_states_actor)
            actions_eval = model_output.actions
            actions_eval_agent_wise = actions_eval.agent_wise
            log_pi_eval = model_output.log_probs

            action_q_1, action_q_2 = self.model.Qpolicy(joint_observations=obs_joint_t,
                                                        joint_actions=actions_joint_t,
                                                        agent_indices=agent_indices_t,
                                                        rnn_states_1=rnn_states_critic_1,
                                                        rnn_states_2=rnn_states_critic_2)

            if self.use_rnn:
                actions_joint_next = self.get_joint_input(actions_eval.agent_wise,
                                                          output_shape=(batch.batch_size, batch.seq_length + 1, -1))
                q_next = self.model.Qtarget(joint_observations=obs_joint,
                                            joint_actions=actions_joint_next,
                                            agent_indices=batch.agent_indices,
                                            rnn_states_1=rnn_states_critic_1,
                                            rnn_states_2=rnn_states_critic_2)
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                log_pi_next = AgentGroupedTensor(
                    {k: v[:, :, 1:] for k, v in log_pi_eval.grouped_tensor.items()}, self.agent_grouping
                )
                log_pi_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in log_pi_eval.grouped_tensor.items()}
                actions_eval_agent_wise = {k: v[:, :-1] for k, v in actions_eval_agent_wise.items()}
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

            # calculate loss and update networks
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])
                agent_keys = self.groups[group]

                # update critic
                action_q_1_i = tf.reshape(action_q_1.packed(group), [-1])
                action_q_2_i = tf.reshape(action_q_2.packed(group), [-1])
                log_pi_next_eval = tf.reshape(log_pi_next.packed(group), [-1])
                alpha = self.current_alpha(group, log_pi_next_eval.dtype)
                rewards = tf.reshape(batch.rewards.packed(group), [-1])
                terminals = tf.reshape(batch.terminals.packed(group), [-1])

                target_value = tf.reshape(q_next.packed(group), [-1]) - alpha * log_pi_next_eval
                backup = rewards + (1 - terminals) * self.gamma * target_value
                backup = tf.stop_gradient(backup)
                td_error_1 = action_q_1_i - backup
                td_error_1 *= mask_values
                td_error_2 = action_q_2_i - backup
                td_error_2 *= mask_values

                loss_c = (tf.reduce_sum(td_error_1 ** 2) + tf.reduce_sum(td_error_2 ** 2)) / tf.reduce_sum(mask_values)

                gradients = tape.gradient(loss_c, self.model.critics[group].trainable_variables)
                if self.use_grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.optimizer[group]['critic'].apply_gradients(
                        zip(gradients, self.model.critics[group].trainable_variables))
                else:
                    self.optimizer[group]['critic'].apply_gradients(
                        zip(gradients, self.model.critics[group].trainable_variables))
                info_train.update({f"{group}/loss_critic": loss_c})

                # update actor
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
                                                            group_key=group,
                                                            rnn_states_1=rnn_states_critic_1,
                                                            rnn_states_2=rnn_states_critic_2)
                log_pi_eval_i = tf.reshape(log_pi_eval.packed(group), [-1])
                policy_q = tf.reshape(tf.minimum(policy_q_1.packed(group), policy_q_2.packed(group)), [-1])

                masked_loss_a = tf.reduce_sum((alpha * log_pi_eval_i - policy_q) * mask_values)
                loss_a = masked_loss_a / tf.reduce_sum(mask_values)

                gradients = tape.gradient(loss_a, self.model.actors[group].trainable_variables)
                if self.use_grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.optimizer[group]['actor'].apply_gradients(
                        zip(gradients, self.model.actors[group].trainable_variables))
                else:
                    self.optimizer[group]['actor'].apply_gradients(
                        zip(gradients, self.model.actors[group].trainable_variables))

                info_train.update({
                    f"{group}/loss_actor": loss_a,
                    f"{group}/predictQ": tf.reduce_mean(policy_q),
                })

                # Automatic entropy tuning
                if self.use_automatic_entropy_tuning:
                    log_alpha = self.alpha_layer[group].log_alpha
                    alpha_loss = -tf.reduce_mean(
                        log_alpha * tf.stop_gradient(log_pi_eval_i + self.target_entropy[group]))
                    gradients = tape.gradient(alpha_loss, self.alpha_layer[group].trainable_variables)
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.alpha_optimizer[group].apply_gradients(
                        zip(gradients, self.alpha_layer[group].trainable_variables))
                    info_train.update({f"{group}/alpha_loss": alpha_loss,
                                       f"{group}/alpha": alpha})
                else:
                    for key in self.group_keys:
                        info_train.update({f"{group}/alpha_loss": tf.Tensor(0.0, dtype=tf.float32),
                                           f"{group}/alpha": self.alpha[key]})

        self.model.soft_update(self.tau)
        return info_train

    def update(self, sample):
        self.iterations += 1

        # Prepare training data.
        batch = self.build_training_data(
            sample,
            use_actions_mask=False
        )

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
