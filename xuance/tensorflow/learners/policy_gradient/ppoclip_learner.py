"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(PPOCLIP_Learner, self).__init__(config, policy, callback)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.mse_loss = tk.losses.MeanSquaredError()
        self.is_continuous = self.policy.is_continuous

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                outputs, mu, std, v_pred = self.policy(obs_batch)
                # calculate log prob
                log_std = tf.math.log(std + 1e-8)
                log_prob = -0.5 * (((act_batch - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
                log_prob_a = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
                # calculate entropy
                entropy = tf.reduce_sum(0.5 + 0.5 * tf.math.log(2.0 * np.pi) + log_std, axis=-1, keepdims=True)
            else:
                outputs, logits, v_pred = self.policy(obs_batch)
                # calculate log prob
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                log_prob_a = tf.gather(log_prob, act_batch, axis=-1, batch_dims=-1)
                # calculate entropy
                probs = tf.exp(log_prob)
                entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=True)

            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob_a - old_logp)
            surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            surrogate2 = adv_batch * ratio
            a_loss = -tf.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
            c_loss = self.mse_loss(ret_batch, v_pred)
            e_loss = tf.reduce_mean(entropy)
            loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return a_loss, c_loss, e_loss, v_pred

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, c_loss, e_loss, v_pred = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, v_pred, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'][:, None], dtype=tf.float32)
        old_logp = tf.convert_to_tensor(samples['aux_batch']['old_logp'][:, None], dtype=tf.float32)
        if self.is_continuous:
            act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        else:
            act_batch = tf.convert_to_tensor(samples["actions"][:, None], dtype=tf.int32)

        a_loss, c_loss, e_loss, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch, old_logp)
        info = {
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy(),
        }

        return info
