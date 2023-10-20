"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *
from xuance.tensorflow.utils.operations import merge_distributions


class MAPPO_KL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.value_clip_range = config.value_clip_range
        super(MAPPO_KL_Learner, self).__init__(config, policy, optimizer, device, modeldir)
        self.target_kl = config.target_kl
        self.kl_coef = 1.0

    def save_model(self):
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        self.policy.actor.save_weights(model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.sort()
            model_path = path + model_names[-1]
            self.policy.actor.load_weights(model_path)
        except:
            raise "Failed to load model! Please train and save the model first."

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            pi_dist_old = sample['pi_dist_old']
            returns = tf.convert_to_tensor(sample['values'])
            advantages = tf.convert_to_tensor(sample['advantages'])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            with tf.GradientTape() as tape:
                inputs = {'obs': obs, 'ids': IDs}
                _, pi_dist = self.policy(inputs)
                log_pi = pi_dist.log_prob(actions)
                old_dist = merge_distributions(pi_dist_old)
                kl = tf.reduce_mean(pi_dist.kl_divergence(old_dist))
                log_pi_old = old_dist.log_prob(actions)
                ratio = tf.reshape(tf.exp(log_pi - log_pi_old), (batch_size, self.n_agents, 1))
                advantages_mask = tf.stop_gradient(advantages) * agent_mask
                loss_a = -tf.reduce_mean(ratio * advantages_mask) + self.kl_coef * kl

                entropy = tf.reshape(pi_dist.entropy(), agent_mask.shape) * agent_mask
                loss_e = tf.reduce_mean(entropy)

                state_expand = tf.tile(tf.expand_dims(state, axis=-2), (1, self.n_agents, 1))
                value = self.policy.values(state_expand, IDs) * agent_mask
                if self.args.use_value_clip:
                    value_clipped = returns + tf.clip_by_value(value-returns, -self.value_clip_range, self.value_clip_range)
                    value_target = advantages_mask + returns * agent_mask
                    loss_v = (value - value_target) ** 2
                    loss_v_clipped = (value_clipped * agent_mask - value_target) ** 2
                    loss_c = tf.reduce_mean(tf.maximum(loss_v, loss_v_clipped))
                else:
                    y_true = tf.reshape(tf.stop_gradient(returns * agent_mask), [-1])
                    y_pred = tf.reshape(value, [-1])
                    loss_c = tk.losses.mean_squared_error(y_true, y_pred)

                loss = loss_a + self.args.vf_coef * loss_c - self.args.ent_coef * loss_e
                gradients = tape.gradient(loss, self.policy.parameters_train)
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_train)
                    if grad is not None
                ])

                if kl > self.target_kl * 1.5:
                    self.kl_coef = self.kl_coef * 2.
                elif kl < self.target_kl * 0.5:
                    self.kl_coef = self.kl_coef / 2.
                self.kl_coef = np.clip(self.kl_coef, 0.1, 20)

            # Logger
            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "actor_loss": loss_a.numpy(),
                "critic_loss": loss_c.numpy(),
                "entropy": loss_e.numpy(),
                "loss": loss.numpy(),
                "predict_value": tf.math.reduce_mean(value).numpy()
            }

            return info
