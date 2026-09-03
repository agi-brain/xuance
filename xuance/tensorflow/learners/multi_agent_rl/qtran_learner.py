"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.learners import OffPolicyMultiAgentLearner
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OffPolicyMARLBatch


class QTRAN_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(QTRAN_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    @tf.function
    def forward_fn(self, **kwargs):
        batch_size = kwargs["batch_size"]
        state = kwargs["global_states"]
        obs = AgentGroupedTensor(kwargs["observations"], self.agent_grouping)
        actions = AgentGroupedTensor(kwargs["actions"], self.agent_grouping)
        rewards = AgentGroupedTensor(kwargs["rewards"], self.agent_grouping)
        terminals = AgentGroupedTensor(kwargs["terminals"], self.agent_grouping)
        agent_mask = AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping)
        agent_indices = AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping)

        if self.use_actions_mask:
            avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
            avail_actions_next = AgentGroupedTensor(kwargs["next_avail_actions"], self.agent_grouping)
        else:
            avail_actions = None
            avail_actions_next = None

        state_next = kwargs["next_global_states"]
        obs_next = AgentGroupedTensor(kwargs["next_observations"], self.agent_grouping)

        rewards_tot = tf.reduce_mean(tf.stack(list(rewards.agent_wise.values()), axis=1), axis=1)
        terminals = tf.cast(tf.stack(list(terminals.agent_wise.values()), axis=1), dtype=tf.bool)
        terminals_tot = tf.cast(tf.reduce_all(terminals, axis=1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            model_output = self.model(observations=obs,
                                      agent_indices=agent_indices,
                                      avail_actions=avail_actions)
            actions_greedy, q_eval = model_output.actions, model_output.values
            hidden_state = {k: v.embeddings for k, v in model_output.rep_out.items()}

            target_model_output = self.model.Qtarget(observations=obs_next,
                                                     agent_indices=agent_indices)
            q_next = target_model_output.values
            hidden_state_next = {k: v.embeddings for k, v in target_model_output.rep_out.items()}

            if self.config.double_q:
                a_next_greedy = self.model(observations=obs_next,
                                           agent_indices=agent_indices,
                                           avail_actions=avail_actions_next).actions
            else:
                a_next_greedy = {}

            q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
            for key in self.agent_keys:
                mask_values = agent_mask.agent_wise[key]
                q_eval_a[key] = tf.reshape(tf.gather(q_eval.agent_wise[key],
                                                     tf.expand_dims(tf.cast(actions.agent_wise[key], dtype=tf.int32),
                                                                    axis=-1), axis=-1, batch_dims=-1), [-1])
                q_eval_greedy_a[key] = tf.reshape(tf.gather(q_eval.agent_wise[key],
                                                            tf.expand_dims(tf.cast(actions_greedy.agent_wise[key],
                                                                                   dtype=tf.int32),
                                                                           axis=-1), axis=-1, batch_dims=-1), [-1])

                if self.use_actions_mask:
                    q_next.agent_wise[key] = tf.where(avail_actions_next[key] == 0,
                                                      tf.cast(-1e10, q_next.agent_wise[key].dtype),
                                                      q_next.agent_wise[key])

                if self.config.double_q:
                    q_next_a[key] = tf.reshape(tf.gather(q_next.agent_wise[key],
                                                         tf.expand_dims(tf.cast(a_next_greedy.agent_wise[key],
                                                                                dtype=tf.int32),
                                                                        axis=-1), axis=-1, batch_dims=-1), [-1])
                else:
                    a_next_greedy[key] = tf.expand_dims(tf.argmax(q_next.agent_wise[key],
                                                                  axis=-1, output_type=tf.int64), axis=-1)
                    q_next_a[key] = tf.reshape(tf.reduce_max(q_next.agent_wise[key], axis=-1), [-1])

                q_eval_a[key] *= mask_values
                q_eval_greedy_a[key] *= mask_values
                q_next_a[key] *= mask_values

            if self.config.agent == "QTRAN_base":
                # -- TD Loss --
                q_joint, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
                a_next_greedy = {k: tf.reshape(v, [batch_size, -1]) for k, v in a_next_greedy.agent_wise.items()}
                q_joint_next, _ = self.model.Q_tran_target(state_next, hidden_state_next,
                                                           AgentGroupedTensor(a_next_greedy, self.agent_grouping),
                                                           agent_mask)

                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
                y_dqn = tf.stop_gradient(y_dqn)
                loss_td = self.mse_loss(y_dqn, q_joint)  # TD loss

                # -- Opt Loss --
                # Argmax across the current agents' actions
                q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
                actions_greedy = {k: tf.reshape(v, [batch_size, -1]) for k, v in actions_greedy.agent_wise.items()}
                q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state,
                                                          AgentGroupedTensor(actions_greedy, self.agent_grouping),
                                                          agent_mask)
                q_joint_greedy_hat = tf.stop_gradient(q_joint_greedy_hat)
                error_opt = q_tot_greedy - q_joint_greedy_hat + v_joint
                loss_opt = tf.reduce_mean(error_opt ** 2)  # Opt loss

                # -- Nopt Loss --
                q_tot = self.model.Q_tot(q_eval_a)
                q_joint_hat = q_joint
                error_nopt = q_tot - tf.stop_gradient(q_joint_hat) + v_joint
                error_nopt = tf.clip_by_value(error_nopt, clip_value_min=-float("inf"), clip_value_max=0.0)
                loss_nopt = tf.reduce_mean(error_nopt ** 2)  # NOPT loss

            elif self.config.agent == "QTRAN_alt":
                # -- TD Loss -- (Computed for all agents)
                q_count, v_joint = self.model.Q_tran(state, hidden_state, actions, agent_mask)
                actions_choosen = tf.stack([actions[k] for k in self.agent_keys], axis=1)
                actions_choosen = tf.reshape(actions_choosen, [-1, self.n_agents, 1])
                q_joint_choosen = tf.reshape(tf.gather(q_count, tf.cast(actions_choosen, dtype=tf.int32),
                                                       axis=-1, batch_dims=-1), [-1, self.n_agents])
                q_next_count, _ = self.model.Q_tran_target(state_next, hidden_state_next, a_next_greedy, agent_mask)
                actions_next_choosen = tf.stack([a_next_greedy[k] for k in self.agent_keys], axis=1)
                actions_next_choosen = tf.reshape(actions_next_choosen, [-1, self.n_agents, 1])
                q_joint_next_choosen = tf.reshape(tf.gather(q_next_count, tf.cast(actions_next_choosen, dtype=tf.int32),
                                                            axis=-1, batch_dims=-1), [-1, self.n_agents])

                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
                y_dqn = tf.stop_gradient(y_dqn)
                loss_td = self.mse_loss(q_joint_choosen, y_dqn)  # TD loss

                # -- Opt Loss -- (Computed for all agents)
                q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
                q_joint_greedy_hat, _ = self.model.Q_tran(state, hidden_state, actions_greedy, agent_mask)
                actions_greedy_current = tf.stack([actions_greedy[k] for k in self.agent_keys], axis=1)
                actions_greedy_current = tf.reshape(actions_greedy_current, [-1, self.n_agents, 1])
                q_joint_greedy_hat_all = tf.reshape(tf.gather(q_joint_greedy_hat, tf.cast(actions_greedy_current,
                                                                                          dtype=tf.int32),
                                                              axis=-1, batch_dims=-1), [-1, self.n_agents])
                q_joint_greedy_hat_all = tf.stop_gradient(q_joint_greedy_hat_all)
                error_opt = q_tot_greedy - q_joint_greedy_hat_all + v_joint
                loss_opt = tf.reduce_mean(error_opt ** 2)  # Opt loss

                # -- Nopt Loss --
                q_eval_count = tf.stack([q_eval[k] for k in self.agent_keys], axis=1)
                q_eval_count = tf.reshape(q_eval_count, [batch_size * self.n_agents, -1])
                q_sums = tf.reshape(tf.stack([q_eval_a[k] for k in self.agent_keys], axis=1), [-1, self.n_agents])
                q_sums_repeat = tf.repeat(tf.expand_dims(q_sums, axis=1), repeats=self.n_agents, axis=1)
                agent_mask_diag = 1.0 - tf.eye(self.n_agents, dtype=tf.float32)
                agent_mask_diag = tf.repeat(tf.expand_dims(agent_mask_diag, axis=0), repeats=batch_size, axis=0)
                q_sum_mask = tf.reduce_sum(q_sums_repeat * agent_mask_diag, axis=-1)
                q_count_for_nopt = tf.stop_gradient(tf.reshape(q_count, [batch_size * self.n_agents, -1]))
                v_joint_repeated = tf.reshape(tf.repeat(v_joint, repeats=self.n_agents, axis=1), [-1, 1])
                error_nopt = q_eval_count + tf.reshape(q_sum_mask, [-1, 1]) - q_count_for_nopt + v_joint_repeated
                error_nopt_min = tf.reduce_min(error_nopt, axis=-1)
                loss_nopt = tf.reduce_mean(error_nopt_min ** 2)  # NOPT loss

            else:
                raise ValueError("Mixer {} not recognised.".format(self.config.agent))

            # calculate the loss function
            loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt

            gradients = tape.gradient(loss, self.model.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))

        return tf.reduce_mean(q_joint), loss_td, loss_opt, loss_nopt, loss

    @tf.function
    def forward_rnn_fn(self, **kwargs):
        batch_size = kwargs["batch_size"]
        seq_len = kwargs["seq_length"]
        state = kwargs["global_states"]
        obs = AgentGroupedTensor(kwargs["observations"], self.agent_grouping)
        actions = AgentGroupedTensor(kwargs["actions"], self.agent_grouping)
        rewards = AgentGroupedTensor(kwargs["rewards"], self.agent_grouping)
        terminals = AgentGroupedTensor(kwargs["terminals"], self.agent_grouping)
        agent_mask = AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping)
        agent_indices = AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping)
        filled = tf.reshape(kwargs["filled_masks"], [-1, 1])
        filled_n = tf.repeat(filled, repeats=self.n_agents, axis=-1)

        if self.use_actions_mask:
            avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
        else:
            avail_actions = None

        rewards_tot = tf.reshape(tf.reduce_mean(tf.stack(list(rewards.agent_wise.values()), axis=1), axis=1), [-1, 1])
        terminals = tf.cast(tf.stack(list(terminals.agent_wise.values()), axis=1), dtype=tf.bool)
        terminals_tot = tf.cast(tf.reshape(tf.reduce_all(terminals, axis=1), [-1, 1]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            rnn_states = self.model.init_rnn_states(batch_size)
            model_output = self.model(observations=obs, agent_indices=agent_indices, avail_actions=avail_actions,
                                      rnn_states=rnn_states)
            actions_greedy, q_eval = model_output.actions, model_output.values
            hidden_state = {k: v.embeddings for k, v in model_output.rep_out.items()}

            target_model_output = self.model.Qtarget(observations=obs, agent_indices=agent_indices,
                                                     rnn_states=rnn_states)
            q_next_seq = target_model_output.values
            hidden_state_next = {k: v.embeddings for k, v in target_model_output.rep_out.items()}

            q_eval_a, q_eval_greedy_a, q_next, q_next_a = {}, {}, {}, {}
            actions_greedy_eval, actions_next_greedy = {}, {}
            for key in self.agent_keys:
                mask_values = agent_mask.agent_wise[key]
                hidden_state[key] = hidden_state[key][:, :-1]
                hidden_state_next[key] = hidden_state_next[key][:, :-1]
                actions_greedy_eval[key] = actions_greedy.agent_wise[key][:, :-1]
                q_eval_a[key] = tf.reshape(tf.gather(q_eval.agent_wise[key][:, :-1],
                                                     tf.expand_dims(tf.cast(actions.agent_wise[key],
                                                                            dtype=tf.int32), axis=-1),
                                                     axis=-1, batch_dims=-1), [batch_size, seq_len])
                q_eval_greedy_a[key] = tf.reshape(tf.gather(q_eval.agent_wise[key][:, :-1],
                                                            tf.expand_dims(
                                                                tf.cast(actions_greedy.agent_wise[key][:, :-1],
                                                                        dtype=tf.int32), axis=-1),
                                                            axis=-1, batch_dims=-1), [batch_size, seq_len])
                q_next[key] = q_next_seq.agent_wise[key][:, 1:]

                if self.use_actions_mask:
                    q_next[key] = tf.where(avail_actions.agent_wise[key][:, 1:] == 0,
                                           tf.cast(-1e10, q_next[key].dtype),
                                           q_next[key])
                if self.config.double_q:
                    act_next = actions_greedy.agent_wise[key][:, 1:]
                    q_next_a[key] = tf.reshape(tf.gather(q_next[key], tf.expand_dims(tf.cast(act_next, dtype=tf.int32),
                                                                                     axis=-1),
                                                         axis=-1, batch_dims=-1), [batch_size, seq_len])
                    actions_next_greedy[key] = act_next
                else:
                    actions_next_greedy[key] = tf.argmax(q_next[key], axis=-1, output_type=tf.int32)
                    q_next_a[key] = tf.reshape(tf.reduce_max(q_next[key], axis=-1, keepdims=True),
                                               [batch_size, seq_len])

                q_eval_a[key] *= mask_values
                q_eval_greedy_a[key] *= mask_values
                q_next_a[key] *= mask_values

            if self.config.agent == "QTRAN_base":
                # -- TD Loss --
                q_joint, v_joint = self.model.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
                q_joint_next, _ = self.model.Q_tran_target(state[:, 1:], hidden_state_next,
                                                           AgentGroupedTensor(actions_next_greedy, self.agent_grouping),
                                                           agent_mask)
                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
                y_dqn = tf.stop_gradient(y_dqn)
                td_error = (q_joint - y_dqn) * filled
                loss_td = tf.reduce_sum(td_error ** 2) / tf.reduce_sum(filled)  # TD loss

                # -- Opt Loss --
                # Argmax across the current agents' actions
                q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
                q_joint_greedy_hat, _ = self.model.Q_tran(state[:, :-1], hidden_state,
                                                          AgentGroupedTensor(actions_greedy_eval, self.agent_grouping),
                                                          agent_mask)
                q_joint_greedy_hat = tf.stop_gradient(q_joint_greedy_hat)
                error_opt = (q_tot_greedy - q_joint_greedy_hat + v_joint) * filled
                loss_opt = tf.reduce_sum(error_opt ** 2) / tf.reduce_sum(filled)  # Opt loss

                # -- Nopt Loss --
                q_tot = self.model.Q_tot(q_eval_a)
                q_joint_hat = q_joint
                error_nopt = q_tot - tf.stop_gradient(q_joint_hat) + v_joint
                error_nopt = tf.clip_by_value(error_nopt, clip_value_min=-float("inf"), clip_value_max=0.0) * filled
                loss_nopt = tf.reduce_sum(error_nopt ** 2) / tf.reduce_sum(filled)  # NOPT loss

            elif self.config.agent == "QTRAN_alt":
                # -- TD Loss -- (Computed for all agents)
                q_count, v_joint = self.model.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
                actions_choosen = tf.stack([actions[k] for k in self.agent_keys], axis=2)
                actions_choosen = tf.reshape(actions_choosen, [-1, self.n_agents, 1])
                q_joint_choosen = tf.reshape(tf.gather(q_count, tf.cast(actions_choosen, dtype=tf.int32),
                                                       axis=-1, batch_dims=-1), [-1, self.n_agents])

                q_next_count, _ = self.model.Q_tran_target(state[:, 1:], hidden_state_next, actions_next_greedy,
                                                           agent_mask)
                actions_next_choosen = tf.stack([actions_next_greedy[k] for k in self.agent_keys], axis=2)

                actions_next_choosen = tf.reshape(actions_next_choosen, [-1, self.n_agents, 1])
                q_joint_next_choosen = tf.reshape(tf.gather(q_next_count, tf.cast(actions_next_choosen, dtype=tf.int32),
                                                            axis=-1, batch_dims=-1), [-1, self.n_agents])

                y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
                y_dqn = tf.stop_gradient(y_dqn)
                td_errors = (q_joint_choosen - y_dqn) * filled_n
                loss_td = tf.reduce_sum(td_errors ** 2) / tf.reduce_sum(filled_n)  # TD loss

                # -- Opt Loss -- (Computed for all agents)
                q_tot_greedy = self.model.Q_tot(q_eval_greedy_a)
                q_joint_greedy_hat, _ = self.model.Q_tran(state[:, :-1], hidden_state, actions_greedy_eval, agent_mask)
                actions_greedy_current = tf.stack([actions_greedy_eval[k] for k in self.agent_keys], axis=2)
                actions_greedy_current = tf.reshape(actions_greedy_current, [-1, self.n_agents, 1])
                q_joint_greedy_hat_all = tf.reshape(tf.gather(q_joint_greedy_hat,
                                                              tf.cast(actions_greedy_current, dtype=tf.int32),
                                                              axis=-1, batch_dims=-1), [-1, self.n_agents])
                q_joint_greedy_hat_all = tf.stop_gradient(q_joint_greedy_hat_all)
                error_opt = (q_tot_greedy - q_joint_greedy_hat_all + v_joint) * filled_n
                loss_opt = tf.reduce_sum(error_opt ** 2) / tf.reduce_sum(filled_n)  # Opt loss

                # -- Nopt Loss --
                q_eval_count = tf.reshape(tf.stack([q_eval[k][:, :-1] for k in self.agent_keys], axis=2),
                                          [batch_size, seq_len, self.n_agents, -1])
                q_eval_count = tf.reshape(q_eval_count, [batch_size * seq_len * self.n_agents, -1])
                q_sums = tf.reshape(tf.stack([q_eval_a[k] for k in self.agent_keys], axis=2),
                                    [-1, seq_len, self.n_agents])
                q_sums = tf.reshape(q_sums, [batch_size * seq_len, self.n_agents])
                q_sums_repeat = tf.repeat(tf.expand_dims(q_sums, axis=1), repeats=self.n_agents, axis=1)
                agent_mask_diag = 1.0 - tf.eye(self.n_agents, dtype=tf.float32)
                agent_mask_diag = tf.repeat(tf.expand_dims(agent_mask_diag, axis=0),
                                            repeats=batch_size * seq_len, axis=0)
                q_sum_mask = tf.reduce_sum(q_sums_repeat * agent_mask_diag, axis=-1)
                q_count_for_nopt = tf.reshape(q_count, [batch_size * seq_len * self.n_agents, -1])
                q_count_for_nopt = tf.stop_gradient(q_count_for_nopt)
                v_joint_repeated = tf.reshape(tf.repeat(v_joint, repeats=self.n_agents, axis=-1), [-1, 1])
                error_nopt = q_eval_count + tf.reshape(q_sum_mask, [-1, 1]) - q_count_for_nopt + v_joint_repeated
                error_nopt_min = tf.reduce_min(error_nopt, axis=-1) * tf.reshape(filled_n, [-1])
                loss_nopt = tf.reduce_sum(error_nopt_min ** 2) / tf.reduce_sum(filled_n)  # NOPT loss
            else:
                raise ValueError("Mixer {} not recognised.".format(self.config.agent))

            # calculate the loss function
            loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt

            gradients = tape.gradient(loss, self.model.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))

        return tf.reduce_mean(q_joint), loss_td, loss_opt, loss_nopt, loss

    @tf.function
    def learn(self, **inputs_learn):
        if self.distributed_training:
            q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.model.mirrored_strategy.run(
                self.forward_fn, kwargs=inputs_learn)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_joint_mean, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_td, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_opt, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_nopt, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_fn(**inputs_learn)

    @tf.function
    def learn_rnn(self, **inputs_learn):
        if self.distributed_training:
            q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.model.mirrored_strategy.run(
                self.forward_rnn_fn, kwargs=inputs_learn)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_joint_mean, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_td, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_opt, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_nopt, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_rnn_fn(**inputs_learn)

    def update(self, sample):
        if self.use_rnn:
            return self.update_rnn(sample)

        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        inputs_learn = {
            "batch_size": batch.batch_size,
            "seq_length": batch.seq_length,
            "global_states": batch.global_states,
            "observations": batch.observations.grouped_tensor,
            "actions": batch.actions.grouped_tensor,
            "rewards": batch.rewards.grouped_tensor,
            "terminals": batch.terminals.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor
            inputs_learn["next_avail_actions"] = batch.next_avail_actions.grouped_tensor

        inputs_learn["next_global_states"] = batch.next_global_states
        inputs_learn["next_observations"] = batch.next_observations.grouped_tensor

        q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.learn(**inputs_learn)

        info.update({
            "Q_joint": q_joint_mean,
            "loss_td": loss_td,
            "loss_opt": loss_opt,
            "loss_nopt": loss_nopt,
            "loss": loss
        })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info

    def update_rnn(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True
        )

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        inputs_learn = {
            "batch_size": batch.batch_size,
            "seq_length": batch.seq_length,
            "global_states": batch.global_states,
            "observations": batch.observations.grouped_tensor,
            "actions": batch.actions.grouped_tensor,
            "rewards": batch.rewards.grouped_tensor,
            "terminals": batch.terminals.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor

        inputs_learn["filled_masks"] = batch.filled_masks

        q_joint_mean, loss_td, loss_opt, loss_nopt, loss = self.learn_rnn(**inputs_learn)

        info.update({
            "Q_joint": q_joint_mean,
            "loss_td": loss_td,
            "loss_opt": loss_opt,
            "loss_nopt": loss_nopt,
            "loss": loss
        })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
