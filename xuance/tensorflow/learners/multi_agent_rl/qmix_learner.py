"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link:
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.learners import OffPolicyMultiAgentLearner
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OffPolicyMARLBatch


class QMIX_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(QMIX_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

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
            # calculate the individual Q values
            rnn_states = self.model.init_rnn_states(batch.batch_size)

            model_output = self.model(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                avail_actions=batch.avail_actions,
                rnn_states=rnn_states
            )
            q_eval = model_output.values

            if self.use_rnn:
                actions_next = model_output.actions

                q_next = self.model.Qtarget(
                    observations=batch.observations,
                    agent_indices=batch.agent_indices,
                    rnn_states=rnn_states
                ).values
                q_eval.grouped_tensor = {k: v[:, :, :-1] for k, v in q_eval.grouped_tensor.items()}
                q_next.grouped_tensor = {k: v[:, :, 1:] for k, v in q_next.grouped_tensor.items()}
                actions_next.grouped_tensor = {k: v[:, :, 1:] for k, v in actions_next.grouped_tensor.items()}

            else:
                q_next = self.model.Qtarget(
                    observations=batch.next_observations,
                    agent_indices=batch.agent_indices,
                ).values

                if self.config.double_q:
                    actions_next = self.model(observations=batch.next_observations,
                                              agent_indices=batch.agent_indices,
                                              avail_actions=batch.next_avail_actions).actions
                else:
                    actions_next = None

            # calculate target values
            q_eval_a, q_next_a = {}, {}
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents),
                                         [batch.batch_size, n_agents, batch.seq_length])

                actions_taken = batch.actions.packed(group)
                q_eval_taken = tf.gather(q_eval.packed(group),
                                         tf.expand_dims(tf.cast(actions_taken, dtype=tf.int32), axis=-1),
                                         axis=-1, batch_dims=-1)
                q_eval_taken = tf.reshape(q_eval_taken, [batch.batch_size, n_agents, batch.seq_length])

                if self.use_actions_mask:
                    if self.use_rnn:
                        next_avail_actions = batch.avail_actions.group(group)[:, 1:]
                    else:
                        next_avail_actions = batch.next_avail_actions.group(group)
                    q_group = q_next.group(group)
                    q_next.grouped_tensor[group] = tf.where(next_avail_actions == 0, tf.cast(-1e10, q_group.dtype),
                                                            q_next.group(group))

                if self.config.double_q:
                    actions_next_taken = tf.expand_dims(actions_next.packed(group), axis=-1)
                    q_next_taken = tf.gather(q_next.packed(group), actions_next_taken, axis=-1, batch_dims=-1)
                    q_next_taken = tf.reshape(q_next_taken, [batch.batch_size, n_agents, batch.seq_length])
                else:
                    q_next_taken = tf.reshape(tf.reduce_max(q_next.packed(group), axis=-1),
                                              [batch.batch_size, n_agents, batch.seq_length])

                q_eval_taken *= mask_values
                q_next_taken *= mask_values

                # get agent-wise values
                for i, agent_key in enumerate(self.groups[group]):
                    q_eval_a[agent_key] = q_eval_taken[:, i]
                    q_next_a[agent_key] = q_next_taken[:, i]

            if self.use_rnn:
                q_tot_eval = tf.reshape(self.model.Q_tot(q_eval_a, batch.global_states[:, :-1]), [-1])
                q_tot_next = tf.reshape(self.model.Qtarget_tot(q_next_a, batch.global_states[:, 1:]), [-1])
            else:
                q_tot_eval = tf.reshape(self.model.Q_tot(q_eval_a, batch.global_states), [-1])
                q_tot_next = tf.reshape(self.model.Qtarget_tot(q_next_a, batch.next_global_states), [-1])

            rewards_tot = tf.reshape(rewards_tot, [-1])
            terminals_tot = tf.reshape(terminals_tot, [-1])
            q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next
            q_tot_target = tf.stop_gradient(q_tot_target)

            # calculate the loss
            if self.use_rnn:
                filled = tf.reshape(batch.filled_masks, [-1])
                td_errors = (q_tot_eval - q_tot_target) * filled
                loss = tf.reduce_sum(td_errors ** 2) / tf.reduce_sum(filled)
            else:
                loss = self.mse_loss(q_tot_eval, q_tot_target)

            gradients = tape.gradient(loss, self.model.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.parameters_model))

        return loss, tf.math.reduce_mean(q_tot_eval)

    @tf.function
    def learn(self, **kwargs):
        if self.distributed_training:
            loss, predictQ = self.model.mirrored_strategy.run(self.forward_fn, kwargs=kwargs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
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
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor
            if not self.use_rnn:
                inputs_learn["next_avail_actions"] = batch.next_avail_actions.grouped_tensor

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks
        else:
            inputs_learn["next_global_states"] = batch.next_global_states
            inputs_learn["next_observations"] = batch.next_observations.grouped_tensor

        loss, q_tot_eval = self.learn(**inputs_learn)

        info.update({
            "loss_Q": loss,
            "predictQ": q_tot_eval
        })

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
