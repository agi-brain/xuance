"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class QTRAN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(QTRAN_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.sync_frequency = config.sync_frequency
        self.mse_loss = tk.losses.MeanSquaredError()

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = tk.optimizers.legacy.Adam(self.config.learning_rate)
        else:
            self.optimizer = tk.optimizers.Adam(self.config.learning_rate)

    # @tf.function
    def forward_fn(self, bs, batch_size, state, obs, actions, rewards_tot, state_next, obs_next, terminals_tot,
                   agent_mask, avail_actions, avail_actions_next, IDs):
        with tf.GradientTape() as tape:
            _, hidden_state, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions)
            _, hidden_state_next, q_next = self.policy.Qtarget(obs_next, agent_ids=IDs)

            q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
            actions_next_greedy = {}
            for key in self.model_keys:
                mask_values = agent_mask[key]
                q_eval_a[key] = tf.reshape(tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                                     axis=-1, batch_dims=-1), [bs])
                q_eval_greedy_a[key] = tf.reshape(tf.gather(q_eval[key], tf.cast(actions_greedy[key][:, None],
                                                                                 dtype=tf.int32),
                                                            axis=-1, batch_dims=-1), [bs])

                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -1e10

                if self.config.double_q:
                    _, _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                    avail_actions=avail_actions, agent_key=key)
                    actions_next_greedy[key] = act_next[key]
                    q_next_a[key] = tf.reshape(tf.gather(q_next[key], act_next[key][:, None],
                                                         axis=-1, batch_dims=-1), [bs])
                else:
                    actions_next_greedy[key] = q_next[key].argmax(dim=-1, keepdim=False)
                    q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

                q_eval_a[key] *= mask_values
                q_eval_greedy_a[key] *= mask_values
                q_next_a[key] *= mask_values

            if self.config.agent == "QTRAN_base":
                # -- TD Loss --
                q_joint, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
                q_joint_next, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy,
                                                            agent_mask)

                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
                loss_td = self.mse_loss(tf.stop_gradient(y_dqn), q_joint)  # TD loss

                # -- Opt Loss --
                # Argmax across the current agents' actions
                q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
                q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
                error_opt = q_tot_greedy - tf.stop_gradient(q_joint_greedy_hat) + v_joint
                loss_opt = tf.reduce_mean(error_opt ** 2)  # Opt loss

                # -- Nopt Loss --
                q_tot = self.policy.Q_tot(q_eval_a)
                q_joint_hat = q_joint
                error_nopt = q_tot - tf.stop_gradient(q_joint_hat) + v_joint
                error_nopt = tf.clip_by_value(error_nopt, clip_value_min=-1e10, clip_value_max=0)
                loss_nopt = tf.reduce_mean(error_nopt ** 2)  # NOPT loss

                q_joint_mean = tf.reduce_mean(q_joint)

            elif self.config.agent == "QTRAN_alt":
                # -- TD Loss -- (Computed for all agents)
                q_count, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
                actions_choosen = itemgetter(*self.model_keys)(actions)
                actions_choosen = tf.reshape(actions_choosen, [-1, self.n_agents, 1])
                q_joint_choosen = tf.reshape(tf.gather(q_count, tf.cast(actions_choosen, dtype=tf.int32),
                                                       axis=-1, batch_dims=-1), [-1, self.n_agents])
                q_next_count, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy,
                                                            agent_mask)
                actions_next_choosen = itemgetter(*self.model_keys)(actions_next_greedy)
                actions_next_choosen = tf.reshape(actions_next_choosen, [-1, self.n_agents, 1])
                q_joint_next_choosen = tf.reshape(tf.gather(q_next_count, tf.cast(actions_next_choosen, dtype=tf.int32),
                                                            axis=-1, batch_dims=-1), [-1, self.n_agents])

                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
                loss_td = self.mse_loss(tf.stop_gradient(y_dqn), q_joint_choosen)  # TD loss

                # -- Opt Loss -- (Computed for all agents)
                q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
                q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
                actions_greedy_current = itemgetter(*self.model_keys)(actions_greedy)
                actions_greedy_current = tf.reshape(actions_greedy_current, [-1, self.n_agents, 1])
                q_joint_greedy_hat_all = tf.reshape(tf.gather(q_joint_greedy_hat, tf.cast(actions_greedy_current,
                                                                                          dtype=tf.int32),
                                                              axis=-1, batch_dims=-1), [-1, self.n_agents])
                error_opt = q_tot_greedy - tf.stop_gradient(q_joint_greedy_hat_all) + v_joint
                loss_opt = tf.reduce_mean(error_opt ** 2)  # Opt loss

                # -- Nopt Loss --
                q_eval_count = tf.reshape(itemgetter(*self.model_keys)(q_eval), [batch_size * self.n_agents, -1])
                q_sums = tf.reshape(itemgetter(*self.model_keys)(q_eval_a), [-1, self.n_agents])
                q_sums_repeat = tf.tile(q_sums[:, None], [1, self.n_agents, 1])
                agent_mask_diag = tf.tile((1 - tf.eye(self.n_agents, dtype=tf.float32))[None], [batch_size, 1, 1])
                q_sum_mask = tf.reduce_sum(q_sums_repeat * agent_mask_diag, axis=-1)
                q_count_for_nopt = tf.reshape(q_count, [batch_size * self.n_agents, -1])
                v_joint_repeated = tf.reshape(tf.tile(v_joint, [1, self.n_agents]), [-1, 1])
                error_nopt = q_eval_count + tf.reshape(q_sum_mask, [-1, 1]) - tf.stop_gradient(q_count_for_nopt) + v_joint_repeated
                error_nopt_min = tf.reduce_min(error_nopt, axis=-1)
                loss_nopt = tf.reduce_mean(error_nopt_min ** 2)  # NOPT loss

                q_joint_mean = tf.reduce_mean(q_joint_choosen)

            else:
                raise ValueError("Mixer {} not recognised.".format(self.config.agent))

            # calculate the loss function
            loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt

            gradients = tape.gradient(loss, self.policy.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))

        return q_joint_mean, loss_td, loss_opt, loss_nopt, loss

    # @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_joint_mean, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_td, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_opt, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_nopt, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = tf.reshape(tf.reduce_mean(rewards[key], axis=1), [batch_size, 1])
            terminals_tot = tf.reshape(tf.reduce_prod(terminals[key], axis=1), [batch_size, 1])
        else:
            bs = batch_size
            rewards_tot = tf.reduce_mean(tf.stack(itemgetter(*self.agent_keys)(rewards), axis=1),
                                         axis=-1, keepdims=True)
            terminals_tot = tf.reduce_prod(tf.stack(itemgetter(*self.agent_keys)(terminals), axis=1),
                                           axis=1, keepdims=True)

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.learn(bs, batch_size, state, obs, actions, rewards_tot,
                                                                      state_next, obs_next, terminals_tot, agent_mask,
                                                                      avail_actions, avail_actions_next, IDs)
        info.update({
            "Q_joint": q_joint_mean.numpy(),
            "loss_td": loss_td.numpy(),
            "loss_opt": loss_opt.numpy(),
            "loss_nopt": loss_nopt.numpy(),
            "loss": loss.numpy()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info,
                                                q_joint_mean=q_joint_mean))

        return info
