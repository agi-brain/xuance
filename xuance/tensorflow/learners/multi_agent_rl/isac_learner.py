"""
Independent Soft Actor-critic (ISAC)
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class ISAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.sync_frequency = sync_frequency
        super(ISAC_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.convert_to_tensor(sample['rewards'])
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                    [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))

            with tf.GradientTape() as tape:
                # calculate the loss function
                inputs = {'obs': obs, 'ids': IDs}
                _, actions_dist = self.policy(inputs)
                actions_eval = actions_dist.sample()
                log_pi_a = tf.expand_dims(actions_dist.log_prob(actions_eval), axis=-1)
                loss_a = self.policy.critic(obs, actions_eval, IDs) - self.alpha * log_pi_a
                loss_a = -tf.reduce_sum(loss_a * agent_mask) / tf.reduce_sum(agent_mask)
                gradients = tape.gradient(loss_a, self.policy.parameters_actor)
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])

            with tf.GradientTape() as tape:
                q_eval = self.policy.critic(obs, actions, IDs)
                actions_next_dist = self.policy.target_actor(obs_next, IDs)
                actions_next = actions_next_dist.sample()
                log_pi_a_next = tf.expand_dims(actions_next_dist.log_prob(actions_next), axis=-1)
                q_next = self.policy.target_critic(obs_next, actions_next, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * (q_next - self.alpha * log_pi_a_next)

                y_true = tf.reshape(tf.stop_gradient(q_target * agent_mask), [-1])
                y_pred = tf.reshape(q_eval * agent_mask, [-1])
                loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])

            self.policy.soft_update(self.tau)

            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic": lr_c.numpy(),
                "loss_actor": loss_a.numpy(),
                "loss_critic": loss_c.numpy(),
                "predictQ": tf.math.reduce_mean(q_eval).numpy()
            }

            return info
