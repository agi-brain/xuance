from xuance.tensorflow.learners import *


class DRQN_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(DRQN_Learner, self).__init__(policy, optimizer, device, modeldir)

    def update(self, obs_batch, act_batch, rew_batch, terminal_batch):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int32)
            rew_batch = tf.convert_to_tensor(rew_batch, dtype=tf.float32)
            ter_batch = tf.convert_to_tensor(terminal_batch, dtype=tf.float32)
            batch_size = obs_batch.shape[0]

            with tf.GradientTape() as tape:
                rnn_hidden = self.policy.init_hidden(batch_size)
                _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
                target_rnn_hidden = self.policy.init_hidden(batch_size)
                _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
                # targetQ = targetQ.max(dim=-1).values

                targetA = F.one_hot(targetA, targetQ.shape[-1])
                targetQ = (targetQ * targetA).sum(dim=-1)

                targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
                predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[-1])).sum(dim=-1)

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
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            info = {
                "Qloss": loss.numpy(),
                "predictQ": predictQ.mean().item(),
                "lr": lr.numpy()
            }

        return info
