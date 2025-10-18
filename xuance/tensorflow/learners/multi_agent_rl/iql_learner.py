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
                 policy: Module,
                 callback):
        super(IQL_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {k: tk.optimizers.legacy.Adam(self.config.learning_rate) for k in self.model_keys}
        else:
            self.optimizer = {k: tk.optimizers.Adam(self.config.learning_rate) for k in self.model_keys}

    @tf.function
    def forward_fn(self, *args):
        bs, obs, actions, rewards, obs_next, terminals, agent_mask, avail_actions, avail_actions_next, IDs = args
        info_train, gradients = {}, {}

        for key in self.model_keys:
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, agent_key=key)
                _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs, agent_key=key)

                q_eval_a = tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                     axis=-1, batch_dims=-1)
                q_eval_a = tf.reshape(q_eval_a, [bs])

                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -1e10

                if self.config.double_q:
                    _, actions_next_greedy, _ = self.policy(obs_next, IDs, agent_key=key, avail_actions=avail_actions)
                    q_next_a = tf.gather(q_next[key], actions_next_greedy[key][:, None], axis=-1, batch_dims=-1)
                    q_next_a = tf.reshape(q_next_a, [bs])
                else:
                    q_next_a = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

                q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a

                # calculate the loss function
                td_error = (q_eval_a - tf.stop_gradient(q_target)) * mask_values
                loss = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(mask_values)

                gradients[key] = tape.gradient(loss, self.policy.parameters_model(key))
                if self.use_grad_clip:
                    gradients[key], _ = tf.clip_by_global_norm(gradients[key], clip_norm=self.grad_clip_norm)
                    self.optimizer[key].apply_gradients(zip(gradients[key], self.policy.parameters_model(key)))
                else:
                    self.optimizer[key].apply_gradients(zip(gradients[key], self.policy.parameters_model(key)))

                info_train.update({
                    f"{key}/loss_Q": loss,
                    f"{key}/predictQ": tf.reduce_mean(q_eval_a)
                })

                info_train.update(self.callback.on_update_agent_wise(self.iterations, key,
                                                                     info=info_train, method="update",
                                                                     mask_values=mask_values, q_eval_a=q_eval_a,
                                                                     q_next_a=q_next_a, q_target=q_target,
                                                                     td_error=td_error, loss=loss))

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
            rewards[key] = tf.reshape(rewards[key], [batch_size * self.n_agents])
            terminals[key] = tf.reshape(terminals[key], [batch_size * self.n_agents])
        else:
            bs = batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        info_train = self.learn(bs, obs, actions, rewards, obs_next, terminals,
                                agent_mask, avail_actions, avail_actions_next, IDs)
        for k, v in info_train.items():
            info_train[k] = v.numpy()
        info.update(info_train)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update_rnn", policy=self.policy, info=info))

        return info
