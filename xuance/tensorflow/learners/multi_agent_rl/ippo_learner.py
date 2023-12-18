"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *
from xuance.tensorflow.utils.operations import update_linear_decay


class IPPO_Learner(LearnerMAS):
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
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.huber_loss = tk.losses.Huber(reduction="none", delta=self.huber_delta)
        super(IPPO_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

    def lr_decay(self, i_step):
        if self.use_linear_lr_decay:
            update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            values = tf.convert_to_tensor(sample['values'])
            returns = tf.convert_to_tensor(sample['values'])
            advantages = tf.convert_to_tensor(sample['advantages'])
            log_pi_old = tf.convert_to_tensor(sample['log_pi_old'])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            with tf.GradientTape() as tape:
                # actor loss
                inputs = {'obs': obs, 'ids': IDs}
                _, pi_dist = self.policy(inputs)
                log_pi = pi_dist.log_prob(actions)
                ratio = tf.reshape(tf.math.exp(log_pi - log_pi_old), [batch_size, self.n_agents, 1])
                advantages_mask = tf.stop_gradient(advantages * agent_mask)
                surrogate1 = ratio * advantages_mask
                surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_mask
                loss_a = -tf.reduce_mean(tf.reduce_sum(tf.minimum(surrogate1, surrogate2), axis=-1))

                # entropy loss
                entropy = tf.reshape(pi_dist.entropy(), agent_mask.shape) * agent_mask
                loss_e = tf.reduce_mean(entropy)

                # critic loss
                _, value_pred = self.policy.get_values(obs, IDs)
                value_pred = tf.expand_dims(value_pred, -1)
                value_target = returns
                if self.use_value_clip:
                    value_clipped = values + tf.clip_by_value(value_pred - values, -self.value_clip_range, self.value_clip_range)
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_target, value_pred)
                        loss_v_clipped = self.huber_loss(value_target, value_clipped)
                    else:
                        loss_v = (value_pred - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c = tf.maximum(loss_v, loss_v_clipped) * tf.squeeze(agent_mask, -1)
                    loss_c = tf.reduce_sum(loss_c) / tf.reduce_sum(agent_mask)
                else:
                    if self.use_huber_loss:
                        loss_v = self.huber_loss(value_pred, value_target) * agent_mask
                    else:
                        loss_v = ((value_pred - value_target) ** 2) * agent_mask
                    loss_c = tf.reduce_sum(loss_v) / tf.reduce_sum(agent_mask)

                loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
                gradients = tape.gradient(loss, self.policy.trainable_param())
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.max_grad_norm), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_param())
                    if grad is not None
                ])

            # Logger
            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "actor_loss": loss_a.numpy(),
                "critic_loss": loss_c.numpy(),
                "entropy": loss_e.numpy(),
                "loss": loss.numpy(),
                "predict_value": tf.math.reduce_mean(value_pred).numpy()
            }

            return info
