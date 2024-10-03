"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(IDDPG_Learner, self).__init__(config, model_keys, agent_keys, policy)
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

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data.
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

        # updata critic
        for key in self.model_keys:
            _, q_eval = self.policy.Qpolicy(observation=obs, actions=actions, agent_ids=IDs)
            _, next_actions = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
            _, q_next = self.policy.Qtarget(next_observation=obs_next, next_actions=next_actions, agent_ids=IDs)
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                q_eval_a = tf.reshape(q_eval[key], [bs])
                q_next_i = tf.reshape(q_next[key], [bs])
                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
                td_error = (q_eval_a - tf.stop_gradient(q_target)) * mask_values
                loss_c = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(mask_values)
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

                info.update({f"{key}/loss_critic": loss_c.numpy(),
                             f"{key}/predictQ": tf.math.reduce_mean(q_eval[key]).numpy()})

        # update actor
        _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
        _, q_policy = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs)
        for key in self.model_keys:
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                q_policy_i = tf.reshape(q_policy[key], [bs])
                loss_a = -tf.reduce_sum(q_policy_i * mask_values) / tf.reduce_sum(mask_values)
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
                info.update({f"{key}/loss_actor": loss_a.numpy()})

        self.policy.soft_update(self.tau)
        return info
