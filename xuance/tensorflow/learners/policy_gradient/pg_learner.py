"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(PG_Learner, self).__init__(config, policy, callback)
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
        self.ent_coef = config.ent_coef
        self.is_continuous = self.policy.is_continuous

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch):
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
                _, logits, _ = self.policy(obs_batch)
                # calculate log prob
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                log_prob_a = tf.gather(log_prob, act_batch, axis=-1, batch_dims=-1)
                # calculate entropy
                probs = tf.exp(log_prob)
                entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=True)

            a_loss = -tf.reduce_mean(ret_batch * log_prob_a)
            e_loss = tf.reduce_mean(entropy)

            loss = a_loss - self.ent_coef * e_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return a_loss, e_loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, e_loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples['returns'][:, None], dtype=tf.float32)
        if self.is_continuous:
            act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        else:
            act_batch = tf.convert_to_tensor(samples["actions"][:, None], dtype=tf.int32)
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch, returns=ret_batch)

        a_loss, e_loss = self.learn(obs_batch, act_batch, ret_batch)

        info.update({
            "actor-loss": a_loss.numpy(),
            "entropy": e_loss.numpy()
        })

        info.update(self.callback.on_update_end(self.iterations, policy=self.policy, info=info,
                                                a_loss=a_loss, e_loss=e_loss))

        return info
