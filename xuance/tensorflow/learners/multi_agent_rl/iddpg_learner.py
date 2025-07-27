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
                 policy: Module,
                 callback):
        super(IDDPG_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.gamma = self.config.gamma
        self.tau = self.config.tau

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {
                key: {'actor': tk.optimizers.legacy.Adam(self.config.learning_rate_actor),
                      'critic': tk.optimizers.legacy.Adam(self.config.learning_rate_critic)}
                for key in self.model_keys}
        else:
            self.optimizer = {
                key: {'actor': tk.optimizers.Adam(self.config.learning_rate_actor),
                      'critic': tk.optimizers.Adam(self.config.learning_rate_critic)}
                for key in self.model_keys}

    @tf.function
    def forward_fn(self, bs, obs, actions, rewards, obs_next, terminals, IDs, agent_mask):
        info_train = {}
        for key in self.model_keys:
            # update critic
            with tf.GradientTape() as tape:
                _, q_eval = self.policy.Qpolicy(observation=obs, actions=actions, agent_ids=IDs, agent_key=key)
                _, next_actions = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs, agent_key=key)
                _, q_next = self.policy.Qtarget(next_observation=obs_next, next_actions=next_actions, agent_ids=IDs,
                                                agent_key=key)
                mask_values = agent_mask[key]
                q_eval_a = tf.reshape(q_eval[key], [bs])
                q_next_i = tf.reshape(q_next[key], [bs])
                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_i
                td_error = (q_eval_a - tf.stop_gradient(q_target)) * mask_values
                loss_c = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(mask_values)
                gradients = tape.gradient(loss_c, self.policy.critic_trainable_variables(key))
                if self.use_grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.optimizer[key]['critic'].apply_gradients(zip(gradients,
                                                                      self.policy.critic_trainable_variables(key)))
                else:
                    self.optimizer[key]['critic'].apply_gradients(zip(gradients,
                                                                      self.policy.critic_trainable_variables(key)))
                info_train.update({f"{key}/loss_critic": loss_c,
                                   f"{key}/predictQ": tf.math.reduce_mean(q_eval[key])})

            # update actor
            with tf.GradientTape() as tape:
                _, actions_eval = self.policy(observation=obs, agent_ids=IDs, agent_key=key)
                _, q_policy = self.policy.Qpolicy(observation=obs, actions=actions_eval, agent_ids=IDs, agent_key=key)

                mask_values = agent_mask[key]
                q_policy_i = tf.reshape(q_policy[key], [bs])
                loss_a = -tf.reduce_sum(q_policy_i * mask_values) / tf.reduce_sum(mask_values)
                gradients = tape.gradient(loss_a, self.policy.actor_trainable_variables(key))
                if self.use_grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                    self.optimizer[key]['actor'].apply_gradients(zip(gradients,
                                                                     self.policy.actor_trainable_variables(key)))
                else:
                    self.optimizer[key]['actor'].apply_gradients(zip(gradients,
                                                                     self.policy.actor_trainable_variables(key)))
                info_train.update({f"{key}/loss_actor": loss_a})
        self.policy.soft_update(self.tau)
        return info_train

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            info_train = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return info_train[0]
        else:
            return self.forward_fn(*inputs)

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
            rewards[key] = tf.reshape(rewards[key], [batch_size * self.n_agents])
            terminals[key] = tf.reshape(terminals[key], [batch_size * self.n_agents])
        else:
            bs = batch_size

        info_train = self.learn(bs, obs, actions, rewards, obs_next, terminals, IDs, agent_mask)
        for k, v in info_train.items():
            info_train[k] = v.numpy()
        info.update(info_train)

        return info
