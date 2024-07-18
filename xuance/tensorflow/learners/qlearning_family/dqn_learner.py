"""
Deep Q-Network (DQN)
Paper link: https://www.nature.com/articles/nature14236
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class DQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(DQN_Learner, self).__init__(config, policy)
        lr_scheduler = tk.optimizers.schedules.ExponentialDecay(config.learning_rate, decay_steps=config.running_steps,
                                                                decay_rate=0.9)
        self.optimizer = tk.optimizers.Adam(lr_scheduler)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        with tf.device(self.device):
            obs_batch = samples['obs']
            act_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.int32)
            next_batch = samples['obs_next']
            rew_batch = tf.convert_to_tensor(samples['rewards'], dtype=tf.float32)
            ter_batch = tf.convert_to_tensor(samples['terminals'], dtype=tf.float32)

            with tf.GradientTape() as tape:
                _, _, evalQ = self.policy(obs_batch)
                _, _, targetQ = self.policy.target(next_batch)
                targetQ = tf.math.reduce_max(targetQ, axis=-1)
                targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
                targetQ = tf.stop_gradient(targetQ)

                predictQ = tf.math.reduce_sum(evalQ * tf.one_hot(act_batch, evalQ.shape[1]), axis=-1)

                loss = tk.losses.mean_squared_error(targetQ, predictQ)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()
            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "Qloss": loss.numpy(),
                "predictQ": tf.math.reduce_mean(predictQ).numpy(),
                "lr": lr.numpy()
            }

        return info
