"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(DRQN_Learner, self).__init__(config, policy)
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
        self.n_actions = self.policy.action_dim

    @tf.function
    def forward_fn(self, batch_size, obs_batch, act_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            rnn_hidden = self.policy.init_hidden(batch_size)
            _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
            target_rnn_hidden = self.policy.init_hidden(batch_size)
            _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
            # targetQ = targetQ.max(dim=-1).values

            targetA = tf.one_hot(targetA, targetQ.shape[-1])
            targetQ = tf.reduce_mean(targetQ * targetA, axis=-1)

            targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
            predictQ = tf.reduce_mean(evalQ * tf.one_hot(act_batch, evalQ.shape[-1]), axis=-1)

            targetQ = tf.reshape(targetQ, [-1])
            predictQ = tf.reshape(predictQ, [-1])
            loss = tk.losses.mean_squared_error(targetQ, predictQ)
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

        return predictQ, loss

    @tf.function
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
        rew_batch = samples['rewards']
        ter_batch = samples['terminals'].astype(np.float32)
        batch_size = obs_batch.shape[0]

        predictQ, loss = self.learn(batch_size, obs_batch, act_batch, rew_batch, ter_batch)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info = {
            "Qloss": loss.numpy(),
            "predictQ": tf.math.reduce_mean(predictQ).numpy()
        }

        return info
