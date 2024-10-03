"""
Distributional Reinforcement Learning (C51DQN)
Paper link: http://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class C51_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(C51_Learner, self).__init__(config, policy)
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
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency

    @tf.function
    def forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            _, _, evalZ = self.policy(obs_batch)
            _, targetA, targetZ = self.policy.target(next_batch)

            current_dist = tf.reduce_sum(evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
            target_dist = tf.stop_gradient(
                tf.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1), axis=1))

            current_supports = self.policy.supports
            next_supports = tf.expand_dims(rew_batch, 1) + self.gamma * self.policy.supports * (
                        1 - tf.expand_dims(ter_batch, 1))
            next_supports = tf.clip_by_value(next_supports, self.policy.v_min, self.policy.v_max)

            projection = 1 - tf.math.abs(
                (tf.expand_dims(next_supports, -1) - tf.expand_dims(current_supports, 0))) / self.policy.deltaz
            target_dist = tf.squeeze(
                tf.linalg.matmul(tf.expand_dims(target_dist, 1), tf.clip_by_value(projection, 0, 1)), 1)

            loss = -tf.reduce_mean(tf.reduce_sum((target_dist * tf.math.log(current_dist + 1e-8)), axis=1))

            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            else:
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

        return loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions'].astype(np.int32)
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']
        loss = self.learn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info = {
            "Qloss": loss.numpy(),
        }

        return info

