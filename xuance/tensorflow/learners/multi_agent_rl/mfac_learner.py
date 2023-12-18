"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class MFAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.tau = config.tau
        super(MFAC_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.optimizer = optimizer

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int32)
            act_mean = tf.convert_to_tensor(sample['act_mean'])
            returns = tf.convert_to_tensor(sample['returns'])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            act_mean_n = tf.tile(tf.expand_dims(act_mean, axis=1), (1, self.n_agents, 1))

            with tf.GradientTape() as tape:
                inputs = {"obs": obs, "ids": IDs}
                _, pi_dist = self.policy(inputs)
                log_pi = pi_dist.log_prob(actions)
                log_pi = tf.expand_dims(log_pi, -1)
                entropy = pi_dist.entropy()
                entropy = tf.expand_dims(entropy, -1)

                targets = returns
                value_pred = self.policy.critic(obs, act_mean_n, IDs)
                advantages = tf.stop_gradient(targets - value_pred)
                td_error = value_pred - tf.stop_gradient(targets)

                pg_loss = -tf.reduce_sum((advantages * log_pi) * agent_mask) / tf.reduce_sum(agent_mask)
                vf_loss = tf.reduce_sum((td_error ** 2) * agent_mask) / tf.reduce_sum(agent_mask)
                entropy_loss = tf.reduce_sum(entropy * agent_mask) / tf.reduce_sum(agent_mask)
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                gradients = tape.gradient(loss, self.policy.trainable_param)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_param)
                    if grad is not None
                ])

            # Logger
            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "pg_loss": pg_loss.numpy(),
                "vf_loss": vf_loss.numpy(),
                "entropy_loss": entropy_loss.numpy(),
                "loss": loss.numpy(),
                "predicted_value": tf.reduce_mean(value_pred).numpy()
            }

            return info
