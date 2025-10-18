"""
Independent Advantage Actor Critic (IAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.utils import ValueNorm
from xuance.tensorflow.learners import LearnerMAS


class IAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(IAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.model_keys}
        else:
            self.value_normalizer = None
        self.is_continuous = self.policy.is_continuous

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.legacy.Adam(self.config.learning_rate)
            else:
                self.optimizer = tk.optimizers.legacy.Adam(self.config.learning_rate)
        else:
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.Adam(self.config.learning_rate)
            else:
                self.optimizer = tk.optimizers.Adam(self.config.learning_rate)

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled, IDs = None, None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)
            actions_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)
            values_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['values']), axis=1)
            returns_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['returns']), axis=1)
            advantages_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['advantages']), axis=1)
            log_pi_old_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['log_pi_old']), axis=1)
            ter_tensor = tf.cast(tf.stack(itemgetter(*self.agent_keys)(sample['terminals']), axis=1), dtype=tf.float32)
            msk_tensor = tf.cast(tf.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), axis=1), dtype=tf.float32)
            if self.use_rnn:
                obs = {k: tf.reshape(obs_tensor, [bs, seq_length, -1])}
                if len(actions_tensor.shape) == 3:
                    actions = {k: tf.reshape(actions_tensor, [bs, seq_length])}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: tf.reshape(actions_tensor, [bs, seq_length, -1])}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: tf.reshape(values_tensor, [bs, seq_length])}
                returns = {k: tf.reshape(returns_tensor, [bs, seq_length])}
                advantages = {k: tf.reshape(advantages_tensor, [bs, seq_length])}
                log_pi_old = {k: tf.reshape(log_pi_old_tensor, [bs, seq_length])}
                terminals = {k: tf.reshape(ter_tensor, [bs, seq_length])}
                agent_mask = {k: tf.reshape(msk_tensor, [bs, seq_length])}
                IDs = tf.reshape(tf.tile(tf.eye(self.n_agents, dtype=np.float32)[None, :, None, :],
                                         [batch_size, 1, seq_length + 1, 1]), [bs, seq_length + 1, self.n_agents])
            else:
                obs = {k: tf.reshape(obs_tensor, [bs, -1])}
                if self.is_continuous:
                    actions = {k: tf.reshape(tf.cast(actions_tensor, dtype=tf.float32), [bs, -1])}
                else:
                    actions = {k: tf.reshape(tf.cast(actions_tensor, dtype=tf.int32), [bs, 1])}
                values = {k: tf.reshape(values_tensor, [bs])}
                returns = {k: tf.reshape(returns_tensor, [bs])}
                advantages = {k: tf.reshape(advantages_tensor, [bs])}
                log_pi_old = {k: tf.reshape(log_pi_old_tensor, [bs])}
                terminals = {k: tf.reshape(ter_tensor, [bs])}
                agent_mask = {k: tf.reshape(msk_tensor, [bs])}
                IDs = tf.reshape(tf.tile(tf.eye(self.n_agents, dtype=np.float32)[None],
                                         [batch_size, 1, 1]), [bs, self.n_agents])

            if use_actions_mask:
                avail_a = tf.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: tf.reshape(avail_a, [bs, seq_length, -1])}
                else:
                    avail_actions = {k: tf.reshape(avail_a, [bs, -1])}

        else:
            obs = {k: tf.convert_to_tensor(sample['obs'][k], dtype=tf.float32) for k in self.agent_keys}
            if self.is_continuous:
                actions = {k: tf.convert_to_tensor(sample['actions'][k], dtype=tf.float32) for k in self.agent_keys}
            else:
                actions = {k: tf.expand_dims(tf.convert_to_tensor(sample['actions'][k], dtype=tf.int32), axis=-1)
                           for k in self.agent_keys}
            values = {k: tf.convert_to_tensor(sample['values'][k], dtype=tf.float32) for k in self.agent_keys}
            returns = {k: tf.convert_to_tensor(sample['returns'][k], dtype=tf.float32) for k in self.agent_keys}
            advantages = {k: tf.convert_to_tensor(sample['advantages'][k], dtype=tf.float32) for k in self.agent_keys}
            log_pi_old = {k: tf.convert_to_tensor(sample['log_pi_old'][k], dtype=tf.float32) for k in self.agent_keys}
            terminals = {k: tf.convert_to_tensor(sample['terminals'][k], dtype=tf.float32) for k in self.agent_keys}
            agent_mask = {k: tf.convert_to_tensor(sample['agent_mask'][k], dtype=tf.float32) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: tf.convert_to_tensor(sample['avail_actions'][k], dtype=tf.float32)
                                 for k in self.agent_keys}

        if use_global_state:
            state = tf.convert_to_tensor(sample['state'], dtype=tf.float32)

        if self.use_rnn:
            filled = tf.convert_to_tensor(sample['filled'], dtype=tf.float32)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'obs': obs,
            'actions': actions,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    # @tf.function
    def forward_fn(self, *args):
        bs, obs, actions, agent_mask, avail_actions, values, returns, advantages, IDs = args
        with tf.GradientTape() as tape:
            loss_a, loss_e, loss_c = [], [], []
            if self.is_continuous:
                _, pi_mu, pi_std = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
                for key in self.model_keys:
                    mask_values = agent_mask[key]
                    mask_values_sum = tf.reduce_sum(mask_values)
                    log_2pi = tf.math.log(2.0 * np.pi)
                    # policy gradient loss
                    log_std = tf.math.log(pi_std[key] + 1e-8)
                    log_prob = -0.5 * (((actions[key] - pi_mu[key]) / (pi_std[key] + 1e-8)) ** 2 + 2.0 * log_std + log_2pi)
                    log_pi = tf.reduce_sum(log_prob, axis=-1, keepdims=False)
                    pg_loss = -tf.reduce_sum((advantages[key] * log_pi) * mask_values) / mask_values_sum
                    loss_a.append(pg_loss)

                    # entropy loss
                    entropy = tf.reduce_sum(0.5 + 0.5 * log_2pi + log_std, axis=-1, keepdims=True)
                    entropy_loss = tf.reduce_sum(entropy * mask_values) / mask_values_sum
                    loss_e.append(entropy_loss)
            else:
                _, pi_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
                for key in self.model_keys:
                    mask_values = agent_mask[key]
                    mask_values_sum = tf.reduce_sum(mask_values)
                    # policy gradient loss
                    log_prob = tf.nn.log_softmax(pi_logits[key], axis=-1)
                    log_pi = tf.gather(log_prob, actions[key], axis=-1, batch_dims=-1)
                    log_pi = tf.squeeze(log_pi, axis=-1)
                    pg_loss = -tf.reduce_sum((advantages[key] * log_pi) * mask_values) / mask_values_sum
                    loss_a.append(pg_loss)

                    # entropy loss
                    probs = tf.exp(log_prob)
                    entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=False)
                    entropy_loss = tf.reduce_sum(entropy * mask_values) / mask_values_sum
                    loss_e.append(entropy_loss)

            _, values_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)
            for key in self.model_keys:
                # value loss
                value_pred_i = tf.reshape(values_pred_dict[key], [bs])
                value_target = tf.reshape(returns[key], [bs])
                values_i = tf.reshape(values[key], [bs])
                if self.use_value_clip:
                    value_clipped = values_i + tf.clip_by_value(value_pred_i - values_i,
                                                                -self.value_clip_range, self.value_clip_range)
                    if self.use_value_norm:
                        self.value_normalizer[key].update(tf.reshape(value_target, [bs, 1]))
                        value_target = tf.reshape(self.value_normalizer[key].normalize(tf.reshape(value_target,
                                                                                                  [bs, 1])), [bs])
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i, self.huber_delta)
                        loss_v_clipped = tk.losses.huber(value_target, value_clipped, self.huber_delta)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c_ = tf.maximum(loss_v, loss_v_clipped) * mask_values
                    loss_c.append(tf.reduce_sum(loss_c_) / mask_values_sum)
                else:
                    if self.use_value_norm:
                        self.value_normalizer[key].update(value_target)
                        value_target = self.value_normalizer[key].normalize(value_target)
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i, self.huber_delta) * mask_values
                    else:
                        loss_v = ((value_pred_i - value_target) ** 2) * mask_values
                    loss_c.append(tf.reduce_sum(loss_v) / mask_values_sum)

            # Total loss
            loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return loss, loss_a, loss_c, loss_e, values_pred_dict

    # @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss, a_loss, c_loss, e_loss, v_pred = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, v_pred, axis=None))
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
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        loss, a_loss, c_loss, e_loss, v_pred = self.learn(bs, obs, actions, agent_mask, avail_actions,
                                                          values, returns, advantages, IDs)

        info.update({f"predict_value/{key}": tf.reduce_mean(v_pred[key]).numpy() for key in self.model_keys})

        info.update({
            # "learning_rate": lr,
            "pg_loss": sum(a_loss).numpy(),
            "vf_loss": sum(c_loss).numpy(),
            "entropy_loss": sum(e_loss).numpy(),
            "loss": loss.numpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
