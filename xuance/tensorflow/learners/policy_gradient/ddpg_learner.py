from xuance.tensorflow.learners import *


class DDPG_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizers: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(DDPG_Learner, self).__init__(policy, optimizers, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch)
            rew_batch = tf.convert_to_tensor(rew_batch)
            ter_batch = tf.convert_to_tensor(terminal_batch)

            # critic update
            with tf.GradientTape() as tape:
                action_q = self.policy.Qaction(obs_batch, act_batch)
                target_q = self.policy.Qtarget(next_batch)
                backup = rew_batch + (1 - ter_batch) * self.gamma * target_q
                y_true = tf.reshape(tf.stop_gradient(backup), [-1])
                y_pred = tf.reshape(action_q, [-1])
                q_loss = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(q_loss, self.policy.critic.trainable_variables)
                self.optimizer[1].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.critic.trainable_variables)
                    if grad is not None
                ])

            # actor update
            with tf.GradientTape() as tape:
                policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = -tf.reduce_mean(policy_q)
                gradients = tape.gradient(p_loss, self.policy.actor.trainable_variables)
                self.optimizer[0].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.actor.trainable_variables)
                    if grad is not None
                ])

            self.policy.soft_update(self.tau)

            actor_lr = self.optimizer[0]._decayed_lr(tf.float32)
            critic_lr = self.optimizer[1]._decayed_lr(tf.float32)

            info = {
                "Qloss": q_loss.numpy(),
                "Ploss": p_loss.numpy(),
                "Qvalue": tf.reduce_mean(action_q).numpy(),
                "actor_lr": actor_lr.numpy(),
                "critic_lr": critic_lr.numpy()
            }

            return info
