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
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.td_lambda = config.td_lambda
        self.sync_frequency = sync_frequency
        self.use_global_state = config.use_global_state
        self.sync_frequency = sync_frequency
        super(COMA_Learner, self).__init__(config, policy, optimizer, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.iterations_actor = self.iterations
        self.iterations_critic = 0

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int32)
            actions_onehot = tf.convert_to_tensor(sample['actions_onehot'])
            targets = tf.squeeze(tf.convert_to_tensor(sample['returns']), -1)
            agent_mask = tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32)
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            with tf.GradientTape() as tape:
                # build critic input
                actions_in = tf.repeat(tf.reshape(tf.expand_dims(actions_onehot, 1), [batch_size, 1, -1]), self.n_agents, 1)
                actions_in_mask = 1 - tf.eye(self.n_agents)
                actions_in_mask = tf.reshape(tf.repeat(tf.reshape(actions_in_mask, [-1, 1]), self.dim_act, 1), [self.n_agents, -1])
                actions_in = actions_in * tf.expand_dims(actions_in_mask, 0)
                if self.use_global_state:
                    state = tf.repeat(tf.expand_dims(state, 1), self.n_agents, 1)
                    critic_in = tf.concat([state, obs, actions_in], axis=-1)
                else:
                    critic_in = tf.concat([obs, actions_in])
                # get critic value
                _, q_eval = self.policy.get_values(critic_in)
                q_eval_a = tf.squeeze(tf.gather(q_eval, tf.expand_dims(actions, -1), axis=-1, batch_dims=-1), -1)
                q_eval_a *= agent_mask
                targets *= agent_mask
                loss_c = tf.reduce_sum((q_eval_a - tf.stop_gradient(targets)) ** 2) / tf.reduce_sum(agent_mask)
                gradients = tape.gradient(loss_c, self.policy.parameters_critic)
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])

            with tf.GradientTape() as tape:
                # calculate baselines
                inputs_policy = {'obs': obs, 'ids': IDs}
                _, pi_probs = self.policy(inputs_policy, epsilon=epsilon)
                baseline = tf.math.reduce_sum(pi_probs * q_eval, axis=-1)
                pi_a = tf.squeeze(tf.gather(pi_probs, tf.expand_dims(actions, -1), axis=-1, batch_dims=-1), -1)
                log_pi_a = tf.math.log(pi_a)
                advantages = tf.stop_gradient(q_eval_a - baseline)
                log_pi_a *= agent_mask
                advantages *= agent_mask
                loss_coma = -tf.reduce_sum(advantages * log_pi_a) / tf.reduce_sum(agent_mask)
                gradients = tape.gradient(loss_coma, self.policy.param_actor())
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.param_actor())
                    if grad is not None
                ])

            # Logger
            lr_a = self.optimizer['actor']._decayed_lr(tf.float32)
            lr_c = self.optimizer['critic']._decayed_lr(tf.float32)

            info = {
                "learning_rate_actor": lr_a.numpy(),
                "learning_rate_critic": lr_c.numpy(),
                "actor_loss": loss_coma.numpy(),
                "critic_loss": loss_c.numpy(),
                "advantage": tf.math.reduce_mean(advantages).numpy()
            }

            return info
