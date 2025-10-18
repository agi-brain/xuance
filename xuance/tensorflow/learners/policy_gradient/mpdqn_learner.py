"""
Multi-pass parameterised deep Q network (MP-DQN)
Paper link: https://arxiv.org/pdf/1905.04388.pdf
Implementation: TensorFlow2
"""
from xuance.common import Sequence
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class MPDQN_Learner(Learner):
    def __init__(self,
                 policy: Module,
                 optimizers: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(MPDQN_Learner, self).__init__(policy, optimizers, device, model_dir)
        self.mse_loss = tk.losses.MeanSquaredError()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            obs_batch = tf.convert_to_tensor(obs_batch)
            disact_batch = tf.convert_to_tensor(act_batch[:, 0], dtype=tf.int32)
            conact_batch = tf.convert_to_tensor(act_batch[:, 1:])
            rew_batch = tf.convert_to_tensor(rew_batch)
            next_batch = tf.convert_to_tensor(next_batch)
            ter_batch = tf.convert_to_tensor(terminal_batch)

            # optimize Q-network
            with tf.GradientTape() as tape:
                target_conact = self.policy.Atarget(next_batch)
                target_q = self.policy.Qtarget(next_batch, target_conact)
                target_q = tf.squeeze(tf.reduce_max(target_q, 1, keepdims=True)[0])

                target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

                eval_qs = self.policy.Qeval(obs_batch, conact_batch)
                eval_q = tf.gather(eval_qs, tf.reshape(disact_batch, [-1, 1]), axis=-1, batch_dims=-1)
                y_true = tf.reshape(tf.stop_gradient(target_q), [-1])
                y_pred = tf.reshape(eval_q, [-1])
                q_loss = self.mse_loss(y_true, y_pred)

                gradients = tape.gradient(q_loss, self.policy.qnetwork.trainable_variables)
                self.optimizer[1].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.qnetwork.trainable_variables)
                    if grad is not None
                ])

            # optimize actor network
            with tf.GradientTape() as tape:
                policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = -tf.reduce_mean(policy_q)
                gradients = tape.gradient(p_loss, self.policy.conactor.trainable_variables)
                self.optimizer[0].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.conactor.trainable_variables)
                    if grad is not None
                ])

            self.policy.soft_update(self.tau)

            self.writer.add_scalar("Q_loss", q_loss.numpy(), self.iterations)
            self.writer.add_scalar("P_loss", q_loss.numpy(), self.iterations)
            self.writer.add_scalar('Qvalue', tf.math.reduce_mean(eval_q).numpy(), self.iterations)
