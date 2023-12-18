"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.temperature = config.temperature
        self.sync_frequency = sync_frequency
        super(MFQ_Learner, self).__init__(config, policy, optimizer, device, model_dir)

    def get_boltzmann_policy(self, q):
        return tf.math.softmax(q / self.temperature, axis=-1)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int32)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            act_mean = tf.convert_to_tensor(sample['act_mean'])
            act_mean_next = tf.convert_to_tensor(sample['act_mean_next'])
            rewards = tf.convert_to_tensor(sample['rewards'])
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), (-1, self.n_agents, 1))
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32), (-1, self.n_agents, 1))
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]
    
            with tf.GradientTape() as tape:
                act_mean = tf.tile(tf.expand_dims(act_mean, axis=1), (1, self.n_agents, 1))
                act_mean_next = tf.tile(tf.expand_dims(act_mean_next, axis=1), (1, self.n_agents, 1))
                inputs = {"obs": obs, "act_mean": act_mean, "ids": IDs}
                _, _, q_eval = self.policy(inputs)
                q_eval_a = tf.gather(q_eval, tf.reshape(actions, (batch_size, self.n_agents, 1)), axis=-1, batch_dims=-1)
                q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
                shape = q_next.shape
                pi = self.get_boltzmann_policy(q_next)
                v_mf = tf.linalg.matmul(tf.reshape(q_next, (-1, 1, shape[-1])),
                                        tf.reshape(tf.expand_dims(pi, axis=-1), (-1, shape[-1], 1)))
                v_mf = tf.reshape(v_mf, shape[0:-1] + (1,))
                q_target = rewards + (1 - terminals) * self.args.gamma * v_mf

                # calculate the loss function
                y_true = tf.reshape(tf.stop_gradient(q_target * agent_mask), [-1])
                y_pred = tf.reshape(q_eval_a, [-1])
                loss = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(loss, self.policy.eval_Qhead.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.eval_Qhead.trainable_variables)
                    if grad is not None
                ])
    
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "loss_Q": loss.numpy(),
                "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
            }

            return info
