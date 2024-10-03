"""
DQN with Quantile Regression (QRDQN)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11791
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class QRDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(QRDQN_Learner, self).__init__(config, policy)
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
            current_quantile = tf.math.reduce_sum(
                evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
            target_quantile = tf.math.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1),
                                                 axis=1)
            target_quantile = tf.expand_dims(rew_batch, 1) + self.gamma * target_quantile * (
                        1 - tf.expand_dims(ter_batch, 1))
            target_quantile = tf.stop_gradient(target_quantile)
            loss = tk.losses.mean_squared_error(tf.reshape(target_quantile, [-1, ]),
                                                tf.reshape(current_quantile, [-1, ]))
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
        return current_quantile, loss

    def learn(self, *inputs):
        if self.distributed_training:
            predictQ, loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions'].astype(np.int32)
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']

        current_quantile, loss = self.learn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info = {
            "Qloss": loss.numpy(),
            "predictQ": tf.math.reduce_mean(current_quantile).numpy()
        }

        return info
