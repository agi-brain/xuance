"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.learners import OffPolicyMultiAgentLearner
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OffPolicyMARLBatch


class WQMIX_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(WQMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}
        self.alpha = config.alpha

    @tf.function
    def forward_fn(self, **kwargs):
        batch = OffPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
            global_states=kwargs["global_states"],
            observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
            actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
            rewards=AgentGroupedTensor(kwargs["rewards"], self.agent_grouping),
            terminals=AgentGroupedTensor(kwargs["terminals"], self.agent_grouping),
            agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
            agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
        )
        if self.use_actions_mask:
            batch.avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)
            if not self.use_rnn:
                batch.next_avail_actions = AgentGroupedTensor(kwargs["next_avail_actions"], self.agent_grouping)
        if self.use_rnn:
            batch.filled_masks = kwargs["filled_masks"]
        else:
            batch.next_global_states = kwargs["next_global_states"]
            batch.next_observations = AgentGroupedTensor(kwargs["next_observations"], self.agent_grouping)

        rewards_tot = tf.reduce_mean(tf.stack(list(batch.rewards.agent_wise.values()), axis=1), axis=1)
        terminals = tf.cast(tf.stack(list(batch.terminals.agent_wise.values()), axis=1), dtype=tf.bool)
        terminals_tot = tf.cast(tf.reduce_all(terminals, axis=1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            ############################################
            #  Feedforward
            ############################################

            # calculate the individual Q value
            rnn_states = self.model.init_rnn_states(batch.batch_size)
            model_output = self.model(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                avail_actions=batch.avail_actions,
                rnn_states=rnn_states
            )
            actions_greedy = model_output.actions
            q_eval = model_output.values

            rnn_states_cent = self.model.init_centralized_rnn_states(batch.batch_size)
            q_eval_centralized = self.model.q_centralized(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                rnn_states=rnn_states_cent
            ).values

            if self.use_rnn:
                q_next_centralized = self.model.target_q_centralized(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states_cent
                ).values

                if not self.config.double_q:
                    q_next = self.model.Qtarget(
                        observations=batch.observations,
                        agent_indices=batch.agent_indices,
                        rnn_states=rnn_states
                    ).values
                    q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                else:
                    q_next = None

                q_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in q_eval.grouped_tensor.items()}
                q_eval_centralized.grouped_tensor = {k: v[:, :, :-1]
                                                     for k, v in q_eval_centralized.grouped_tensor.items()}
                q_next_centralized.grouped_tensor = {k: v[:, :, 1:]
                                                     for k, v in q_next_centralized.grouped_tensor.items()}
                next_actions_greedy = AgentGroupedTensor({k: v[:, :, 1:]
                                                          for k, v in actions_greedy.grouped_tensor.items()},
                                                         self.agent_grouping)
                actions_greedy.grouped_tensor = {k: v[:, :, :-1] for k, v in actions_greedy.grouped_tensor.items()}

            else:
                q_next_centralized = self.model.target_q_centralized(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices
                ).values

                if self.config.double_q:
                    a_next_greedy = self.model(
                        observations=batch.next_observations,
                        agent_indices=batch.agent_indices,
                        avail_actions=batch.next_avail_actions
                    ).actions
                    next_actions_greedy = a_next_greedy
                    q_next = None
                else:
                    q_next = self.model.Qtarget(
                        observations=batch.next_observations,
                        agent_indices=batch.agent_indices
                    ).values

                    if self.use_actions_mask:
                        for group in self.group_keys:
                            next_avail_actions = batch.next_avail_actions.group(group)
                            q_group = q_next.group(group)
                            q_next.grouped_tensor[group] = tf.where(next_avail_actions == 0,
                                                                    tf.cast(-1e10, q_group.dtype),
                                                                    q_next.group(group))
                    next_actions_greedy = None

            q_eval_a, q_eval_centralized_a, q_next_centralized_a = {}, {}, {}
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents),
                                         [batch.batch_size, n_agents, batch.seq_length])

                actions_taken = batch.actions.group(group)
                q_eval_taken = tf.gather(q_eval.group(group),
                                         tf.expand_dims(tf.cast(actions_taken, dtype=tf.int32), axis=-1),
                                         axis=-1, batch_dims=-1)
                q_eval_taken = tf.reshape(q_eval_taken, [batch.batch_size, n_agents, batch.seq_length])

                actions_greedy_taken = actions_greedy.group(group)
                q_eval_centralized_taken = tf.gather(q_eval_centralized.group(group),
                                                     tf.expand_dims(tf.cast(actions_greedy_taken, dtype=tf.int32),
                                                                    axis=-1), axis=-1, batch_dims=-1)
                q_eval_centralized_taken = tf.reshape(q_eval_centralized_taken,
                                                      [batch.batch_size, n_agents, batch.seq_length])

                if self.config.double_q:
                    actions_next_taken = tf.expand_dims(next_actions_greedy.group(group), axis=-1)
                else:
                    actions_next_taken = tf.expand_dims(tf.argmax(q_next.group(group), axis=-1, output_type=tf.int64),
                                                        axis=-1)
                    actions_next_taken = tf.expand_dims(actions_next_taken, axis=-1)

                q_next_centralized_taken = tf.reshape(tf.gather(q_next_centralized.group(group), actions_next_taken,
                                                                axis=-1, batch_dims=-1),
                                                      [batch.batch_size, n_agents, batch.seq_length])

                q_eval_taken *= mask_values
                q_eval_centralized_taken *= mask_values
                q_next_centralized_taken *= mask_values

                # get agent-wise values
                for i, agent_key in enumerate(self.groups[group]):
                    q_eval_a[agent_key] = q_eval_taken[:, i]
                    q_eval_centralized_a[agent_key] = q_eval_centralized_taken[:, i]
                    q_next_centralized_a[agent_key] = q_next_centralized_taken[:, i]

            ############################################
            #  Calculate total values and loss
            ############################################
            if self.use_rnn:
                state_input = tf.reshape(batch.global_states[:, :-1], [batch.batch_size * batch.seq_length, -1])
                state_input_next = tf.reshape(batch.global_states[:, 1:], [batch.batch_size * batch.seq_length, -1])
            else:
                state_input = batch.global_states
                state_input_next = batch.next_global_states
            # calculate Q_tot
            q_tot_eval = tf.reshape(self.model.Q_tot(individual_values=q_eval_a, states=state_input), [-1])
            # calculate centralized Q
            q_tot_centralized = tf.reshape(self.model.q_feedforward(individual_values=q_eval_centralized_a,
                                                                    states=state_input), [-1])
            # calculate y_i
            q_tot_next_centralized = tf.reshape(self.model.target_q_feedforward(individual_values=q_next_centralized_a,
                                                                                states=state_input_next), [-1])
            rewards_tot = tf.reshape(rewards_tot, [-1])
            terminals_tot = tf.reshape(terminals_tot, [-1])

            target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
            target_value = tf.stop_gradient(target_value)
            td_error = q_tot_eval - target_value

            # calculate the weights
            ones = tf.ones_like(td_error)
            w = ones * self.alpha
            if self.config.agent == "CWQMIX":
                condition_1_list = []
                for group, agent_keys in self.groups.items():
                    n_agents = self.n_group_agents[group]
                    mask_values = tf.reshape(batch.valid_mask(group, n_agents),
                                             [batch.batch_size, n_agents, batch.seq_length])
                    a_greedy = actions_greedy.group(group)
                    act = tf.reshape(batch.actions[group], [batch.batch_size, n_agents, batch.seq_length])
                    condition_1_list.append(tf.logical_and(tf.equal(a_greedy, act), tf.cast(mask_values, tf.bool)))
                condition_1 = tf.reshape(tf.reduce_all(tf.concat(condition_1_list, axis=1), axis=1), [-1])
                condition_2 = target_value > q_tot_centralized
                conditions = tf.logical_or(condition_1, condition_2)
                w = tf.where(conditions, ones, w)
            elif self.config.agent == "OWQMIX":
                condition = td_error < 0
                w = tf.where(condition, ones, w)
            else:
                raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

            # calculate the loss
            if self.use_rnn:
                filled = tf.reshape(batch.filled_masks, [-1])
                loss_central = tf.reduce_sum((((q_tot_centralized - target_value) ** 2) * filled)) / tf.reduce_sum(
                    filled)
                loss_qmix = tf.reduce_sum((tf.stop_gradient(w) * (td_error ** 2) * filled)) / tf.reduce_sum(filled)
            else:
                loss_central = self.mse_loss(target_value, q_tot_centralized)
                loss_qmix = tf.reduce_mean(tf.stop_gradient(w) * (td_error ** 2))

            loss = loss_qmix + loss_central

            # update the networks
            gradients = tape.gradient(loss, self.model.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))

        return loss_qmix, loss_central, loss, tf.math.reduce_mean(q_tot_eval)

    @tf.function
    def learn(self, **kwargs):
        if self.distributed_training:
            loss_qmix, loss_central, loss, predictQ = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_qmix, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_central, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None))
        else:
            return self.forward_fn(**kwargs)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(
            sample=sample,
            use_actions_mask=self.use_actions_mask,
            use_global_state=True,
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
            inputs_learn["avail_actions"] = (
                batch.avail_actions.grouped_tensor
            )
            if not self.use_rnn:
                inputs_learn["next_avail_actions"] = (
                    batch.next_avail_actions.grouped_tensor
                )

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks
        else:
            inputs_learn["next_global_states"] = batch.next_global_states
            inputs_learn["next_observations"] = batch.next_observations.grouped_tensor

        loss_qmix, loss_central, loss, predictQ = self.learn(**inputs_learn)

        info.update({
            "loss_Qmix": loss_qmix,
            "loss_central": loss_central,
            "loss": loss,
            "predictQ": predictQ
        })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
