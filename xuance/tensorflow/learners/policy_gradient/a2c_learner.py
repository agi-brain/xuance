"""
Advantage Actor-Critic (A2C)
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(A2C_Learner, self).__init__(config, policy, callback)
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
        self.mse_loss = tk.losses.MeanSquaredError()
        self.is_continuous = self.policy.is_continuous

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                outputs, mu, std, v_pred = self.policy(obs_batch)
                log_2pi = tf.math.log(2.0 * np.pi)
                # calculate log prob
                log_std = tf.math.log(std + 1e-8)
                log_prob = -0.5 * (((act_batch - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + log_2pi)
                log_prob_a = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
                # calculate entropy
                entropy = tf.reduce_sum(0.5 + 0.5 * log_2pi + log_std, axis=-1, keepdims=True)
            else:
                outputs, logits, v_pred = self.policy(obs_batch)
                # calculate log prob
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                log_prob_a = tf.gather(log_prob, act_batch, axis=-1, batch_dims=-1)
                # calculate entropy
                probs = tf.exp(log_prob)
                entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=True)

            a_loss = -tf.reduce_mean(adv_batch * log_prob_a)
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
        if self.is_continuous:
            act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        else:
            act_batch = tf.convert_to_tensor(samples["actions"][:, None], dtype=tf.int32)

        a_loss, c_loss, e_loss, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch)

        info = {
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy()
        }

        return info
