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
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -policy.action_space[self.agent_keys[0]].shape[-1]
            self.alpha_layer = AlphaLayer(policy.action_dim)
            self.alpha = tf.exp(self.alpha_layer.log_alpha)
            if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
                self.alpha_optimizer = tk.optimizers.legacy.Adam(config.learning_rate_actor)
            else:
                self.alpha_optimizer = tk.optimizers.Adam(config.learning_rate_actor)

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
        with tf.GradientTape() as tape:
            _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
            _, _, action_q_1, action_q_2 = self.policy.Qaction(observation=obs, actions=actions, agent_ids=IDs)
            _, _, next_q = self.policy.Qtarget(next_observation=obs_next, next_actions=actions_next, agent_ids=IDs)

            for key in self.model_keys:
                mask_values = agent_mask[key]
                action_q_1_i, action_q_2_i = action_q_1[key].reshape(bs), action_q_2[key].reshape(bs)
                log_pi_next_eval = log_pi_next[key].reshape(bs)
                next_q_i = next_q[key].reshape(bs)
                target_value = next_q_i - self.alpha * log_pi_next_eval
                backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
                td_error_1, td_error_2 = action_q_1_i - backup.detach(), action_q_2_i - backup.detach()
                td_error_1 *= mask_values
                td_error_2 *= mask_values
                loss_c = ((td_error_1 ** 2).sum() + (td_error_2 ** 2).sum()) / mask_values.sum()
                gradients = tape.gradient(loss_c, self.policy.parameters_critic(key))
                if self.use_grad_clip:
                    self.optimizer['critic'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.parameters_critic(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer['critic'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.parameters_critic(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_critic": loss_c.numpy()})

        # Update actor
        with tf.GradientTape() as tape:
            _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs)
            for key in self.model_keys:
                _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs,
                                                                   agent_key=key)
                log_pi_eval_i = log_pi_eval[key].reshape(bs)
                policy_q = tf.reshape(tf.math.minimum(policy_q_1[key], policy_q_2[key]), bs)
                loss_a = tf.reduce_sum((self.alpha * log_pi_eval_i - policy_q) * mask_values) / tf.reduce_sum(
                    mask_values)
                gradients = tape.gradient(loss_a, self.policy.parameters_actor(key))
                if self.use_grad_clip:
                    self.optimizer[key]['actor'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.parameters_actor(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer[key]['actor'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.parameters_actor(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_actor": loss_a.item(),
                             f"{key}/predictQ": policy_q.mean().item()})

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            with tf.GradientTape() as tape:
                alpha_loss = -(self.log_alpha * (log_pi_eval_i + self.target_entropy).detach()).mean()
                gradients = tape.gradient(alpha_loss, self.alpha_layer.trainable_variables)
                self.alpha_optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.alpha_layer.trainable_variables)
                    if grad is not None
                ])
        else:
            alpha_loss = 0
        info.update({"alpha_loss": alpha_loss.numpy(),
                     "alpha": self.alpha.numpy()})
        self.policy.soft_update(self.tau)
        return info
