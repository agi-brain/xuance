"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class COMA_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.td_lambda = config.td_lambda
        self.sync_frequency = sync_frequency
        super(COMA_Learner, self).__init__(config, policy, optimizer, device, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.iterations_actor = self.iterations
        self.iterations_critic = 0

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

    def build_td_lambda(self, rewards, terminated, agent_mask, target_q_a, max_step_len):
        rewards = rewards.numpy()
        terminated = terminated.numpy()
        agent_mask = agent_mask.numpy()
        target_q_a = target_q_a.numpy()
        returns = np.zeros_like(target_q_a)
        returns[:, -1] = target_q_a[:, -1] * (1 - terminated.sum(dim=1))
        for t in range(max_step_len - 2, -1, -1):
            returns[:, t] = self.td_lambda * self.gamma * returns[:, t + 1] + (rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_q_a[:, t + 1] * (1 - terminated[:, t])) * agent_mask[:, t]
        return tf.convert_to_tensor(returns[:, 0:-1])

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            state_repeat = tf.tile(tf.expand_dims(state, axis=-2), (1, 1, self.n_agents, 1))
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int32)
            actions_onehot = tf.convert_to_tensor(sample['actions_onehot'])
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards'][:, :-1]), axis=-2)
            terminals = tf.convert_to_tensor(sample['terminals'], tf.float32)
            agent_mask = tf.convert_to_tensor(sample['agent_mask'], tf.float32)
            batch_size, step_len = obs.shape[0], obs.shape[1]
            IDs = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(self.n_agents), axis=0), axis=0), multiples=(batch_size, step_len, 1, 1))

            # train critic network

            target_critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs)
            target_q_eval = self.policy.target_critic(target_critic_in)
            target_q_a = tf.gather(target_q_eval, tf.expand_dims(actions, axis=-1), axis=-1, batch_dims=-1)
            target_q_a = tf.reshape(target_q_a, (batch_size, step_len, self.n_agents))
            targets = self.build_td_lambda(rewards, terminals, agent_mask, target_q_a, step_len)

            loss_c_item = 0.0
            q_eval = np.zeros(target_q_eval.shape)[:, :-1]
            for t in reversed(range(step_len-1)):
                with tf.GradientTape() as tape:
                    agent_mask_t = agent_mask[:, t:t+1]
                    actions_t = tf.expand_dims(actions[:, t], axis=-2)
                    critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs, t)
                    q_eval_t = self.policy.critic(critic_in)
                    q_eval[:, t:t+1] = q_eval_t
                    q_eval_a_t = tf.gather(q_eval_t, tf.expand_dims(actions_t, axis=-1), axis=-1, batch_dims=-1)
                    q_eval_a_t = tf.reshape(q_eval_a_t, (batch_size, 1, self.n_agents))
                    q_eval_a_t *= agent_mask_t
                    target_t = targets[:, t:t+1]

                    self.iterations_critic += 1
                    y_true = tf.reshape(tf.stop_gradient(target_t), [-1])
                    y_pred = tf.reshape(q_eval_a_t, [-1])
                    loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                    gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                    self.optimizer['critic'].apply_gradients([
                        (tf.clip_by_norm(grad, self.args.clip_grad), var)
                        for (grad, var) in zip(gradients, self.policy.parameters_critic)
                        if grad is not None
                    ])
                    if self.iterations_critic % self.sync_frequency == 0:
                        self.policy.copy_target()
                    loss_c_item += loss_c
            loss_c_item /= (step_len - 1)
            q_eval = tf.convert_to_tensor(q_eval, dtype=tf.float32)

            # calculate baselines
            with tf.GradientTape() as tape:
                inputs = {'obs': obs, 'ids': IDs}
                _, pi_dist = self.policy(inputs)
                pi = tf.math.softmax(pi_dist.logits, axis=-1)[:, :-1]
                pi_log_prob = pi_dist.log_prob(actions)[:, :-1]
                baseline = tf.reduce_sum(pi * q_eval, axis=-1)

                q_eval_a = tf.gather(q_eval, tf.expand_dims(actions[:, :-1], axis=-1), axis=-1, batch_dims=-1)
                q_eval_a = tf.reshape(q_eval_a, [batch_size, step_len-1, self.n_agents])
                advantages = tf.stop_gradient(q_eval_a - baseline)

                self.iterations_actor += 1
                loss_coma = -tf.reduce_sum((advantages * pi_log_prob) * agent_mask[:, :-1]) / tf.reduce_sum(agent_mask[:, :-1])
                gradients = tape.gradient(loss_coma, self.policy.parameters_actor)
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])

            # Logger
            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic": lr_c.numpy(),
                "actor_loss": loss_coma.numpy(),
                "critic_loss": loss_c_item.numpy(),
                "advantage": tf.math.reduce_mean(advantages).numpy()
            }

            return info
