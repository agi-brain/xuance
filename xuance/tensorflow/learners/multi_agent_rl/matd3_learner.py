"""
Multi-Agent TD3

"""
from xuance.tensorflow.learners import *


class MATD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: Sequence[tk.optimizers.Optimizer],
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100,
                 delay: int = 3
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.delay = delay
        self.sync_frequency = sync_frequency
        super(MATD3_Learner, self).__init__(config, policy, optimizer, device, model_dir)
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

            # train critic
            with tf.GradientTape() as tape:
                _, action_q = self.policy.Qaction(obs, actions, IDs)
                inputs_next = {"obs": obs_next, "ids": IDs}
                actions_next = self.policy.target_actor(inputs_next)
                _, target_q = self.policy.target_critic(obs_next, actions_next, IDs)
                q_target = rewards + (1 - terminals) * self.args.gamma * target_q
                y_pred = tf.reshape(action_q * agent_mask, [-1])
                q_target = tf.tile(q_target, (1, 1, 2))
                y_true = tf.reshape(tf.stop_gradient(q_target * agent_mask), [-1])
                loss_c = tk.losses.mean_squared_error(y_true, y_pred)
                gradients = tape.gradient(loss_c, self.policy.critic_parameters)
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.critic_parameters)
                    if grad is not None
                ])

            # actor update
            with tf.GradientTape() as tape:
                if self.iterations % self.delay == 0:
                    inputs = {"obs": obs, "ids": IDs}
                    _, actions_eval = self.policy(inputs)
                    _, policy_q = self.policy.critic(obs, actions_eval, IDs)
                    p_loss = -tf.reduce_mean(policy_q)
                    gradients = tape.gradient(p_loss, self.policy.actor_net.trainable_variables)
                    self.optimizer['actor'].apply_gradients([
                        (tf.clip_by_norm(grad, self.args.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.actor_net.trainable_variables)
                        if grad is not None
                    ])
                    self.policy.soft_update(self.tau)

            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic_A": lr_c.numpy(),
                "loss_critic": loss_c.numpy(),
                "predictQ": tf.math.reduce_mean(action_q).numpy()
            }
            if self.iterations % self.delay == 0:
                info["loss_actor"] = p_loss.numpy()

            return info
