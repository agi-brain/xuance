"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: TensorFlow 2.X
"""
from operator import itemgetter
from argparse import Namespace
from xuance.common import List, Optional
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(MFQ_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.policy_type = self.policy.policy_type

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {k: tk.optimizers.legacy.Adam(self.config.learning_rate) for k in self.model_keys}
        else:
            self.optimizer = {k: tk.optimizers.Adam(self.config.learning_rate) for k in self.model_keys}

    def build_actions_mean_input(self, sample: Optional[dict], use_parameter_sharing: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        actions_mean, actions_mean_next = None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.n_agents == 1:
                actions_mean_tensor = tf.convert_to_tensor(sample['actions_mean'][k][:, None])
            else:
                actions_mean_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['actions_mean']), axis=1)
            if self.use_rnn:
                actions_mean = {k: tf.reshape(actions_mean_tensor, [bs, seq_length + 1, -1])}
            else:
                actions_mean = {k: tf.reshape(actions_mean_tensor, [bs, -1])}
                if self.n_agents == 1:
                    actions_mean_next_tensor = tf.convert_to_tensor(sample['actions_mean_next'][k][:, None])
                else:
                    actions_mean_next_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['actions_mean_next']), 1)
                actions_mean_next = {k: tf.reshape(actions_mean_next_tensor, [bs, -1])}
        else:
            actions_mean = {k: tf.convert_to_tensor(sample['actions_mean'][k]) for k in self.agent_keys}
            if not self.use_rnn:
                actions_mean_next = {k: tf.convert_to_tensor(sample['actions_mean_next'][k]) for k in self.agent_keys}

        return actions_mean, actions_mean_next

    @tf.function
    def forward_fn(self, *args):
        bs, obs, actions, act_mean, rewards, obs_next, act_mean_next, terminals, agent_mask, avail_actions, avail_actions_next, IDs = args
        info_train, gradients = {}, {}

        with tf.GradientTape(persistent=True) as tape:
            _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, actions_mean=act_mean,
                                       avail_actions=avail_actions)
            _, q_next = self.policy.Qtarget(observation=obs_next, actions_mean=act_mean_next, agent_ids=IDs)

            for key in self.model_keys:
                mask_values = agent_mask[key]
                q_eval_a = tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                     axis=-1, batch_dims=-1)

                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -1e10

                if self.policy_type == "Boltzmann":
                    pi_probs = tf.nn.softmax(q_next[key] / self.policy.temperature)
                    v_mf = tf.reshape(tf.reduce_sum(pi_probs * q_next[key], axis=-1), [-1])
                    q_target = rewards[key] + (1 - terminals[key]) * self.gamma * v_mf
                elif self.policy_type == "greedy":
                    _, actions_next_greedy, _ = self.policy(obs_next, IDs, actions_mean=act_mean_next, agent_key=key,
                                                            avail_actions=avail_actions)
                    q_next_a = tf.reshape(tf.gather(q_next[key], tf.cast(actions_next_greedy[key][:, None],
                                                                         dtype=tf.int32), axis=-1, batch_dims=-1), [bs])
                    q_target = rewards[key] + (1 - terminals[key]) * self.gamma * q_next_a
                else:
                    raise NotImplementedError

                # calculate the loss function
                q_target = tf.stop_gradient(q_target)
                td_error = (q_eval_a - q_target) * mask_values
                loss = tf.reduce_sum((td_error ** 2)) / tf.reduce_sum(mask_values)

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
        act_mean, act_mean_next = self.build_actions_mean_input(sample=sample,
                                                                use_parameter_sharing=self.use_parameter_sharing)
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

        info = self.callback.on_update_start(self.iterations, method="update", policy=self.policy)

        info_train = self.learn(bs, obs, actions, act_mean, rewards, obs_next, act_mean_next,
                                terminals, agent_mask, avail_actions, avail_actions_next, IDs)
        for k, v in info_train.items():
            info_train[k] = v.numpy()
        info.update(info_train)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
