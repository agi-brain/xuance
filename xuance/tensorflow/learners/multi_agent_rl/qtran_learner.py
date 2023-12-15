"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: TensorFlow 2.X
"""
from xuance.tensorflow.learners import *


class QTRAN_Learner(LearnerMAS):
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
        self.sync_frequency = sync_frequency
        super(QTRAN_Learner, self).__init__(config, policy, optimizer, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
            actions_onehot = self.onehot_action(actions, self.dim_act)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'].all(axis=-1, keepdims=True), dtype=tf.float32), [-1, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32),
                                    [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            with tf.GradientTape() as tape:
                inputs_policy = {"obs": obs, "ids": IDs}
                hidden_n, _, q_eval = self.policy(inputs_policy)
                # get mask input
                actions_mask = tf.tile(agent_mask, multiples=(1, 1, self.dim_act))
                hidden_mask = tf.tile(agent_mask, multiples=(1, 1, hidden_n.shape[-1]))
                q_joint, v_joint = self.policy.qtran_net(hidden_n * hidden_mask,
                                                         actions_onehot * actions_mask)
                inputs_target = {"obs": obs_next, "ids": IDs}
                hidden_n_next, q_next_eval = self.policy.target_Q(inputs_target)
                if self.args.double_q:
                    inputs_target = {"obs": obs_next, "ids": IDs}
                    _, actions_next_greedy, _ = self.policy(inputs_target)
                else:
                    actions_next_greedy = tf.argmax(q_next_eval, axis=-1)
                q_joint_next, _ = self.policy.target_qtran_net(hidden_n_next * hidden_mask,
                                                               self.onehot_action(actions_next_greedy,
                                                                                  self.dim_act) * actions_mask)
                y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next
                y_dqn = tf.stop_gradient(tf.reshape(y_dqn, [-1]))
                q_joint = tf.reshape(q_joint, [-1])
                loss_td = tk.losses.mean_squared_error(y_dqn, q_joint)

                action_greedy = tf.argmax(q_eval, axis=-1)  # \bar{u}
                q_eval_greedy_a = tf.gather(q_eval, tf.reshape(action_greedy, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                q_tot_greedy = self.policy.q_tot(q_eval_greedy_a * agent_mask)
                q_joint_greedy_hat, _ = self.policy.qtran_net(hidden_n * hidden_mask,
                                                              self.onehot_action(action_greedy, self.dim_act) * actions_mask)
                error_opt = q_tot_greedy - tf.stop_gradient(q_joint_greedy_hat) + v_joint
                loss_opt = tf.reduce_mean(error_opt ** 2)

                q_eval_a = tf.gather(q_eval, tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                if self.args.agent == "QTRAN_base":
                    q_tot = self.policy.q_tot(q_eval_a * agent_mask)
                    q_joint_hat, _ = self.policy.qtran_net(hidden_n * hidden_mask,
                                                           actions_onehot * actions_mask)
                    error_nopt = q_tot - tf.stop_gradient(q_joint_hat) + v_joint
                    error_nopt = tf.clip_by_value(error_nopt, clip_value_min=-1e10, clip_value_max=0)
                    loss_nopt = tf.reduce_mean(error_nopt ** 2)
                elif self.args.agent == "QTRAN_alt":
                    q_tot_counterfactual = self.policy.qtran_net.counterfactual_values(q_eval, q_eval_a) * actions_mask
                    q_joint_hat_counterfactual = self.policy.qtran_net.counterfactual_values_hat(hidden_n * hidden_mask,
                                                                                                 actions_onehot * actions_mask)
                    v_joint_repeat = tf.tile(tf.expand_dims(v_joint, axis=-1), multiples=(1, self.n_agents, self.dim_act))
                    error_nopt = q_tot_counterfactual - tf.stop_gradient(q_joint_hat_counterfactual) + v_joint_repeat
                    error_nopt_min = tf.reduce_min(error_nopt, axis=-1)
                    loss_nopt = tf.reduce_mean(error_nopt_min ** 2)
                else:
                    raise ValueError("Mixer {} not recognised.".format(self.args.agent))

                # calculate the loss function
                loss = loss_td + self.args.lambda_opt * loss_opt + self.args.lambda_nopt * loss_nopt
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.optimizer._decayed_lr(tf.float32)

                info = {
                    "learning_rate": lr.numpy(),
                    "loss_td": loss_td.numpy(),
                    "loss_opt": loss_opt.numpy(),
                    "loss_nopt": loss_nopt.numpy(),
                    "loss": loss.numpy(),
                    "predictQ": tf.math.reduce_mean(q_eval_a).numpy()
                }

                return info
