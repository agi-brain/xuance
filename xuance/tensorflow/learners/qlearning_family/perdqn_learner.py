"""
DQN with Prioritized Experience Replay (PER-DQN)
Paper link: https://arxiv.org/pdf/1511.05952.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class PerDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(PerDQN_Learner, self).__init__(config, policy, callback)
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
        self.mse_loss = tk.losses.MeanSquaredError()

    @tf.function
    def forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            _, _, evalQ = self.policy(obs_batch)
            _, _, targetQ = self.policy.target(next_batch)
            targetQ = tf.math.reduce_max(targetQ, axis=-1)
            targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
            targetQ = tf.stop_gradient(targetQ)
            predictQ = tf.math.reduce_sum(evalQ * tf.one_hot(act_batch, evalQ.shape[1]), axis=-1)

            td_error = targetQ - predictQ
            loss = self.mse_loss(targetQ, predictQ)
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
        return td_error, predictQ, loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            td_error, predictQ, loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, td_error, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None),
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
        td_error, predictQ, loss = self.learn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info = {
            "Qloss": loss.numpy(),
            "predictQ": tf.math.reduce_mean(predictQ).numpy()
        }

        return np.abs(td_error.numpy()), info
