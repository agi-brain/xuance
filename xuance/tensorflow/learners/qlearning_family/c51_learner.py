from xuance.tensorflow.learners import *


class C51_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(C51_Learner, self).__init__(policy, optimizer, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.cast(tf.convert_to_tensor(act_batch), dtype=tf.int64)
            rew_batch = tf.convert_to_tensor(rew_batch)
            ter_batch = tf.convert_to_tensor(terminal_batch)

            with tf.GradientTape() as tape:
                _, _, evalZ = self.policy(obs_batch)
                _, targetA, targetZ = self.policy.target(next_batch)

                current_dist = tf.reduce_sum(evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
                target_dist = tf.stop_gradient(tf.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1), axis=1))

                current_supports = self.policy.supports
                next_supports = tf.expand_dims(rew_batch, 1) + self.gamma * self.policy.supports * (1 - tf.expand_dims(ter_batch, 1))
                next_supports = tf.clip_by_value(next_supports, self.policy.vmin, self.policy.vmax)

                projection = 1 - tf.math.abs((tf.expand_dims(next_supports, -1) - tf.expand_dims(current_supports, 0))) / self.policy.deltaz
                target_dist = tf.squeeze(tf.linalg.matmul(tf.expand_dims(target_dist, 1), tf.clip_by_value(projection, 0, 1)), 1)

                loss = -tf.reduce_mean(tf.reduce_sum((target_dist * tf.math.log(current_dist + 1e-8)), axis=1))

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
                "lr": lr.numpy(),
            }

            return info

