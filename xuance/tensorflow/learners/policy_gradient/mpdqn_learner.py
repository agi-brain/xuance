from xuance.tensorflow.learners import *

class MPDQN_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizers: Sequence[tk.optimizers.Optimizer],
                 summary_writer: Optional[SummaryWriter] = None,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(MPDQN_Learner, self).__init__(policy, optimizers, summary_writer, device, modeldir)

    def save_model(self):
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        model_path_qnet = model_path + "/qnet"
        model_path_actor = model_path + "/conactor"
        self.policy.qnetwork.save(model_path_qnet)
        self.policy.conactor.save(model_path_actor)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.remove('obs_rms.npy')
            model_names.sort()
            model_path_qnet = path + model_names[-1] + "/qnet"
            model_path_actor = path + model_names[-1] + "/conactor"
            self.policy.qnetwork = tk.models.load_model(model_path_qnet, compile=False)
            self.policy.conactor = tk.models.load_model(model_path_actor, compile=False)
        except:
            raise "Failed to load model! Please train and save the model first."

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
                q_loss = tk.losses.mean_squared_error(y_true, y_pred)

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
