"""
Independent Q-learning (IQL)
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OffPolicyMARLBatch
from xuance.tensorflow.learners import OffPolicyMultiAgentLearner


class IQL_Learner(OffPolicyMultiAgentLearner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(IQL_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.individual_q_networks[k].action_space.n for k in self.group_keys}

    def build_optimizer(self):
        # quite different from xuance.torch, where there each group has its own optimizer
        self.optimizer = keras.optimizers.Adam(self.config.learning_rate)

    @tf.function
    def forward_fn(self, **kwargs):

        info_train, gradients = {}, {}

        batch = OffPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
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
            batch.next_observations = AgentGroupedTensor(kwargs["next_observations"], self.agent_grouping)

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

            # calculate losses and update networks for each group of agents
            loss_group = []
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                rewards = tf.reshape(batch.rewards.packed(group), [-1])
                terminals = tf.reshape(batch.terminals.packed(group), [-1])

                actions_taken = batch.actions.packed(group)
                q_eval_taken = tf.gather(q_eval.packed(group),
                                         tf.expand_dims(tf.cast(actions_taken, dtype=tf.int32), axis=-1),
                                         axis=-1, batch_dims=-1)
                q_eval_taken = tf.reshape(q_eval_taken, [-1])

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
                    q_next_taken = tf.reshape(q_next_taken, [-1])
                else:
                    q_next_taken = tf.reshape(tf.reduce_max(q_next.packed(group), axis=-1), [-1])

                q_target = rewards + (1 - terminals) * self.gamma * q_next_taken
                q_target = tf.stop_gradient(q_target)

                # calculate the loss function
                td_error = (q_eval_taken - q_target) * mask_values
                masked_mean_loss_group = tf.reduce_sum(td_error ** 2) / (tf.reduce_sum(mask_values))
                loss_group.append(masked_mean_loss_group)

                info_train.update({
                    f"{group}/loss_Q": masked_mean_loss_group,
                    f"{group}/predictQ": tf.reduce_mean(q_eval_taken)
                })

            loss = sum(loss_group)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return info_train

    @tf.function
    def learn(self, **kwargs):
        if self.distributed_training:
            info_train = self.model.mirrored_strategy.run(self.forward_fn, kwargs=kwargs)
            return info_train[0]
        else:
            return self.forward_fn(**kwargs)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask)

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        inputs_learn = {
            "batch_size": batch.batch_size,
            "seq_length": batch.seq_length,
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
            inputs_learn["next_observations"] = batch.next_observations.grouped_tensor

        info_train = self.learn(**inputs_learn)

        info.update(info_train)

        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
