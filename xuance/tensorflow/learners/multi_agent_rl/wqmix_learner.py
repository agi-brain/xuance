"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.alpha = config.alpha
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(WQMIX_Learner, self).__init__(config, policy, optimizer, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            state_next = tf.convert_to_tensor(sample['state_next'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'].all(axis=-1, keepdims=True), dtype=tf.float32), [-1, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                    [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            with tf.GradientTape() as tape:
                # calculate Q_tot
                inputs_policy = {"obs": obs, "ids": IDs}
                _, action_max, q_eval = self.policy(inputs_policy)
                action_max = tf.expand_dims(action_max, axis=-1)
                q_eval_a = tf.gather(q_eval, indices=tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)

                # calculate centralized Q
                q_eval_centralized = tf.gather(self.policy.q_centralized(inputs_policy), action_max, axis=-1, batch_dims=-1)
                q_tot_centralized = self.policy.q_feedforward(q_eval_centralized*agent_mask, state)

                # calculate y_i
                inputs_target = {"obs": obs_next, "ids": IDs}
                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(inputs_target)
                    action_next_greedy = tf.expand_dims(action_next_greedy, axis=-1)
                else:
                    q_next_eval = self.policy.target_Q(inputs_target)
                    action_next_greedy = tf.argmax(q_next_eval, axis=-1)
                q_eval_next_centralized = tf.gather(self.policy.target_q_centralized(inputs_target), action_next_greedy, axis=-1, batch_dims=-1)
                q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized*agent_mask, state_next)

                target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized
                td_error = q_tot_eval - tf.stop_gradient(target_value)

                # calculate weights
                ones = tf.ones_like(td_error)
                w = ones * self.alpha
                if self.args.agent == "CWQMIX":
                    condition_1 = tf.cast((action_max == tf.reshape(actions, [-1, self.n_agents, 1])), dtype=tf.float32)
                    condition_1 = tf.reduce_all(tf.cast(condition_1 * agent_mask, dtype=tf.bool), axis=1)
                    condition_2 = target_value > q_tot_centralized
                    conditions = condition_1 | condition_2
                    w = tf.where(conditions, ones, w)
                elif self.args.agent == "OWQMIX":
                    condition = td_error < 0
                    w = tf.where(condition, ones, w)
                else:
                    AttributeError("You have assigned an unexpected WQMIX learner!")

                # calculate losses and train
                y_true = tf.stop_gradient(tf.reshape(target_value, [-1]))
                y_pred = tf.reshape(q_tot_centralized, [-1])
                loss_central = tk.losses.mean_squared_error(y_true, y_pred)
                loss_qmix = tf.reduce_mean((w * (td_error ** 2)))
                loss = loss_qmix + loss_central
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
                "learning_rate": lr.numpy(),
                "loss_Qmix": loss_qmix.numpy(),
                "loss_central": loss_central.numpy(),
                "loss": loss.numpy(),
                "predictQ": tf.math.reduce_mean(q_tot_eval).numpy()
            }

            return info
