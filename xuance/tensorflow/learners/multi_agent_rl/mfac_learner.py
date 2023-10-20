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
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.tau = config.tau
        super(MFAC_Learner, self).__init__(config, policy, optimizer, device, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }

    def save_model(self):
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        self.policy.actor_net.save(model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.sort()
            model_path = path + model_names[-1]
            print(model_path)
            # self.policy = tk.models.load_model(model_path, compile=False)
            self.policy.actor_net.load_weights(model_path)
        except:
            raise "Failed to load model! Please train and save the model first."

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int32)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            act_mean = tf.convert_to_tensor(sample['act_mean'])
            act_mean_n = tf.tile(tf.expand_dims(act_mean, axis=1), (1, self.n_agents, 1))
            # act_mean_next = tf.convert_to_tensor(sample['act_mean_next'])
            rewards = tf.convert_to_tensor(sample['rewards'])
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], tf.float32), (-1, self.n_agents, 1))
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            # train critic network
            with tf.GradientTape() as tape:
                target_pi_dist_next = self.policy.target_actor(obs_next, IDs)
                target_pi_next = tf.math.softmax(target_pi_dist_next.logits, axis=-1)
                actions_next = target_pi_dist_next.stochastic_sample()
                actions_next_onehot = self.onehot_action(actions_next, self.dim_act)
                act_mean_next = tf.reduce_mean(actions_next_onehot, axis=-2, keepdims=False)
                act_mean_n_next = tf.tile(tf.expand_dims(act_mean_next, axis=1), (1, self.n_agents, 1))

                q_eval = self.policy.critic(obs, act_mean_n, IDs)
                q_eval_a = tf.gather(q_eval, tf.reshape(actions, [batch_size, self.n_agents, 1]), batch_dims=-1, axis=-1)

                q_eval_next = self.policy.target_critic(obs_next, act_mean_n_next, IDs)
                shape = q_eval_next.shape
                v_mf = tf.linalg.matmul(tf.reshape(q_eval_next, [-1, 1, shape[-1]]),
                                        tf.reshape(target_pi_next, [-1, shape[-1], 1]))
                v_mf = tf.reshape(v_mf, shape[0:-1] + (1, ))
                q_target = rewards + (1 - terminals) * self.args.gamma * v_mf
                y_true = tf.reshape(tf.stop_gradient(q_target * agent_mask), [-1])
                y_pred = tf.reshape(q_eval_a * agent_mask, [-1])
                loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                self.optimizer['critic'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])

            # train actor network
            with tf.GradientTape() as tape:
                inputs = {'obs': obs, 'ids': IDs}
                _, pi_dist = self.policy(inputs)
                actions_ = pi_dist.stochastic_sample()
                advantages = self.policy.target_critic(obs, act_mean_n, IDs)
                advantages = tf.gather(advantages, tf.reshape(actions_, [batch_size, self.n_agents, 1]))
                log_pi_prob = tf.expand_dims(pi_dist.log_prob(actions_), axis=-1)
                advantages = log_pi_prob * tf.stop_gradient(advantages)
                loss_a = -tf.reduce_sum(advantages) / tf.reduce_sum(agent_mask)
                gradients = tape.gradient(loss_c, self.policy.parameters_actor)
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])

            self.policy.soft_update(self.tau)

            # Logger
            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic": lr_c.numpy(),
                "actor_loss": loss_a.numpy(),
                "critic_loss": loss_c.numpy(),
            }

            return info
