"""
Independent Soft Actor-critic (ISAC)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS
from xuance.tensorflow.learners.policy_gradient.sac_learner import AlphaLayer


class ISAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(ISAC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {
                key: {'actor': tk.optimizers.legacy.Adam(config.learning_rate_actor),
                      'critic': tk.optimizers.legacy.Adam(config.learning_rate_critic)}
                for key in self.model_keys}
        else:
            self.optimizer = {
                key: {'actor': tk.optimizers.Adam(config.learning_rate_actor),
                      'critic': tk.optimizers.Adam(config.learning_rate_critic)}
                for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = {key: config.alpha for key in self.model_keys}
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = {key: -policy.action_space[key].shape[-1] for key in self.model_keys}
            self.alpha_layer = {key: AlphaLayer() for key in self.model_keys}
            self.alpha = {key: tf.exp(self.alpha_layer[key].log_alpha) for key in self.model_keys}
            if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
                self.alpha_optimizer = {key: tk.optimizers.legacy.Adam(config.learning_rate_actor)
                                        for key in self.model_keys}
            else:
                self.alpha_optimizer = {key: tk.optimizers.Adam(config.learning_rate_actor) for key in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # Prepare training data.
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        # Update critic
        _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
        _, _, action_q_1, action_q_2 = self.policy.Qaction(observation=obs, actions=actions, agent_ids=IDs)
        _, _, next_q = self.policy.Qtarget(next_observation=obs_next, next_actions=actions_next, agent_ids=IDs)

        for key in self.model_keys:
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                action_q_1_i, action_q_2_i = tf.reshape(action_q_1[key], [bs]), tf.reshape(action_q_2[key], [bs])
                log_pi_next_eval = tf.reshape(log_pi_next[key], [bs])
                next_q_i = tf.reshape(next_q[key], [bs])
                target_value = next_q_i - self.alpha[key] * log_pi_next_eval
                backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
                backup = tf.stop_gradient(backup)
                td_error_1, td_error_2 = action_q_1_i - backup, action_q_2_i - backup
                td_error_1 *= mask_values
                td_error_2 *= mask_values
                loss_c = (tf.reduce_sum(td_error_1 ** 2) + tf.reduce_sum(td_error_2 ** 2)) / tf.reduce_sum(mask_values)
                gradients = tape.gradient(loss_c, self.policy.critic_trainable_variables(key))
                if self.use_grad_clip:
                    self.optimizer[key]['critic'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.critic_trainable_variables(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer[key]['critic'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.critic_trainable_variables(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_critic": loss_c.numpy()})

        # Update actor
        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs)
        log_pi_eval_i = {}
        for key in self.model_keys:
            with tf.GradientTape() as tape:
                _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs,
                                                                   agent_key=key)
                log_pi_eval_i[key] = tf.reshape(log_pi_eval[key], [bs])
                policy_q = tf.reshape(tf.math.minimum(policy_q_1[key], policy_q_2[key]), [bs])
                loss_a = tf.reduce_sum((self.alpha[key] * log_pi_eval_i[key] - policy_q) * mask_values) / tf.reduce_sum(
                    mask_values)
                gradients = tape.gradient(loss_a, self.policy.actor_trainable_variables(key))
                if self.use_grad_clip:
                    self.optimizer[key]['actor'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.actor_trainable_variables(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer[key]['actor'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.actor_trainable_variables(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_actor": loss_a.numpy(),
                             f"{key}/predictQ": tf.math.reduce_mean(policy_q).numpy()})

        # Automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            for key in self.model_keys:
                with tf.GradientTape() as tape:
                    alpha_loss = -tf.math.reduce_mean(self.alpha_layer[key].log_alpha.value() * (log_pi_eval_i[key] + self.target_entropy[key]))
                    gradients = tape.gradient(alpha_loss, self.alpha_layer[key].trainable_variables)
                    self.alpha_optimizer[key].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.alpha_layer[key].trainable_variables)
                        if grad is not None
                    ])
                    self.alpha[key] = tf.math.exp(self.alpha_layer[key].log_alpha)
                    info.update({f"{key}/alpha_loss": alpha_loss.numpy(),
                                 f"{key}/alpha": self.alpha[key].numpy()})
        else:
            for key in self.model_keys:
                info.update({f"{key}/alpha_loss": 0.0,
                             f"{key}/alpha": self.alpha[key]})

        self.policy.soft_update(self.tau)
        return info
