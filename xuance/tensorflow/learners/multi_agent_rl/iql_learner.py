"""
Independent Q-learning (IQL)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {k: tk.optimizers.legacy.Adam(config.learning_rate) for k in self.model_keys}
        else:
            self.optimizer = {k: tk.optimizers.Adam(config.learning_rate) for k in self.model_keys}
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size

        with tf.GradientTape() as tape:
            _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

            for key in self.model_keys:
                q_eval_a = tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                     axis=-1, batch_dims=-1)
                q_eval_a = tf.reshape(q_eval_a, [bs])

                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -9999999

                if self.config.double_q:
                    _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
                    q_next_a = tf.gather(q_next[key], actions_next_greedy[key][:, None], axis=-1, batch_dims=-1)
                    q_next_a = tf.reshape(q_next_a, [bs])
                else:
                    q_next_a = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

                # calculate the loss function
                td_error = (q_eval_a - tf.stop_gradient(q_target)) * agent_mask[key]
                loss = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(agent_mask[key])

                gradients = tape.gradient(loss, self.policy.parameters_model[key])
                if self.use_grad_clip:
                    self.optimizer[key].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.parameters_model[key])
                        if grad is not None
                    ])
                else:
                    self.optimizer[key].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.parameters_model[key])
                        if grad is not None
                    ])

                info.update({
                    f"{key}/loss_Q": loss.numpy(),
                    f"{key}/predictQ": tf.reduce_mean(q_eval_a).numpy()
                })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
