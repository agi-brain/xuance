from xuance.tensorflow.learners import *


class PerDQN_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(PerDQN_Learner, self).__init__(policy, optimizer, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int32)
            rew_batch = tf.convert_to_tensor(rew_batch)
            ter_batch = tf.convert_to_tensor(terminal_batch)

            with tf.GradientTape() as tape:
                _, _, evalQ = self.policy(obs_batch)
                _, _, targetQ = self.policy.target(next_batch)
                targetQ = tf.math.reduce_max(targetQ, axis=-1)
                targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
                targetQ = tf.stop_gradient(targetQ)
                predictQ = tf.math.reduce_sum(evalQ * tf.one_hot(act_batch, evalQ.shape[1]), axis=-1)

                td_error = targetQ - predictQ
                loss = tk.losses.mean_squared_error(targetQ, predictQ)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            # hard update for target network
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "Qloss": loss.numpy(),
                "predictQ": tf.math.reduce_mean(predictQ).numpy(),
                "lr": lr.numpy()
            }

            return np.abs(td_error.numpy()), info
