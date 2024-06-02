"""
Value Decomposition Actor-Critic (VDAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
from xuance.tensorflow.learners import *
from xuance.torch.utils.operations import update_linear_decay


class VDAC_Learner(LearnerMAS):
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
        self.use_grad_clip, self.grad_clip_norm = config.use_grad_clip, config.grad_clip_norm
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        super(VDAC_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

    def lr_decay(self, i_step):
        if self.use_linear_lr_decay:
            update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

    def update(self, sample):
        info = {}
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            returns = tf.reduce_mean(tf.convert_to_tensor(sample['values']), axis=1)
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            with tf.GradientTape() as tape:
                inputs = {'obs': obs, 'ids': IDs}
                _, pi_dist, value_pred = self.policy(inputs, state=state)
                log_pi = tf.expand_dims(pi_dist.log_prob(actions), -1)
                entropy = tf.reshape(pi_dist.entropy(), agent_mask.shape) * agent_mask

                targets = returns
                advantages = tf.expand_dims(tf.stop_gradient(targets - value_pred), -1)
                td_error = tf.expand_dims(value_pred - tf.stop_gradient(targets), -1)

                pg_loss = -tf.reduce_sum((advantages * log_pi) * agent_mask) / tf.reduce_sum(agent_mask)
                vf_loss = tf.reduce_sum((td_error ** 2) * agent_mask) / tf.reduce_sum(agent_mask)
                entropy_loss = tf.reduce_sum(entropy * agent_mask) / tf.reduce_sum(agent_mask)
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                gradients = tape.gradient(loss, self.policy.trainable_param())
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_param())
                    if grad is not None
                ])

            # Logger
            lr = self.optimizer._decayed_lr(tf.float32)

            info.update({
                "learning_rate": lr.numpy(),
                "pg_loss": pg_loss.numpy(),
                "vf_loss": vf_loss.numpy(),
                "entropy_loss": entropy_loss.numpy(),
                "loss": loss.numpy(),
                "predict_value": tf.reduce_mean(value_pred).numpy()
            })

            return info
