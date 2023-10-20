from xuance.tensorflow.learners import *


class QRDQN_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(QRDQN_Learner, self).__init__(policy, optimizer, summary_writer, device, modeldir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int64)
            rew_batch = tf.convert_to_tensor(rew_batch)
            ter_batch = tf.convert_to_tensor(terminal_batch)

            with tf.GradientTape() as tape:
                _, _, evalZ, _ = self.policy(obs_batch)
                _, targetA, _, targetZ = self.policy(next_batch)
                current_quantile = tf.math.reduce_sum(evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
                target_quantile = tf.math.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1), axis=1)
                target_quantile = tf.expand_dims(rew_batch, 1) + self.gamma * target_quantile * (1 - tf.expand_dims(ter_batch, 1))
                target_quantile = tf.stop_gradient(target_quantile)
                loss = tk.losses.mean_squared_error(tf.reshape(target_quantile, [-1, ]), tf.reshape(current_quantile, [-1, ]))
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
            self.writer.add_scalar("Qloss", loss.numpy(), self.iterations)
            self.writer.add_scalar("predictQ", tf.math.reduce_mean(current_quantile).numpy(), self.iterations)
            self.writer.add_scalar("lr", lr.numpy(), self.iterations)
