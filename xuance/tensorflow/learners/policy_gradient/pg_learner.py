from xuance.tensorflow.learners import *


class PG_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 ent_coef: float = 0.005,
                 clip_grad: Optional[float] = None):
        super(PG_Learner, self).__init__(policy, optimizer, device, model_dir)
        self.ent_coef = ent_coef
        self.clip_grad = clip_grad

    def update(self, obs_batch, act_batch, ret_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch, dtype=tf.float32)
            ret_batch = tf.convert_to_tensor(ret_batch)

            with tf.GradientTape() as tape:
                outputs, _ = self.policy(obs_batch)
                a_dist = self.policy.actor.dist
                log_prob = a_dist.log_prob(act_batch)

                a_loss = -tf.reduce_mean(ret_batch * log_prob)
                e_loss = tf.reduce_mean(a_dist.entropy())

                loss = a_loss - self.ent_coef * e_loss
                gradients = tape.gradient(loss, self.policy.trainable_variables)

                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "actor-loss": a_loss.numpy(),
                "entropy": e_loss.numpy(),
                "learning_rate": lr.numpy()
            }

            return info
