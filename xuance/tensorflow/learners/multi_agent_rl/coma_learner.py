"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from tensorflow import one_hot
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.rl_models.modules import AgentGroupedTensor, OnPolicyMARLBatch
from xuance.tensorflow.learners.multi_agent_rl.iac_learner import IAC_Learner


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, agent_grouping, model, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.model.critics.action_space[k].n for k in self.agent_keys}

    def build_optimizer(self):
        self.optimizer = {
            'actor': keras.optimizers.Adam(self.config.learning_rate_actor),
            'critic': keras.optimizers.Adam(self.config.learning_rate_critic)
        }

    @tf.function
    def forward_fn(self, **kwargs):

        info_train, gradients = {}, {}

        batch = OnPolicyMARLBatch(
            batch_size=kwargs["batch_size"],
            seq_length=kwargs["seq_length"],
            global_states=kwargs["global_states"],
            observations=AgentGroupedTensor(kwargs["observations"], self.agent_grouping),
            actions=AgentGroupedTensor(kwargs["actions"], self.agent_grouping),
            returns=AgentGroupedTensor(kwargs["returns"], self.agent_grouping),
            agent_masks=AgentGroupedTensor(kwargs["agent_masks"], self.agent_grouping),
            agent_indices=AgentGroupedTensor(kwargs["agent_indices"], self.agent_grouping),
        )
        if self.use_actions_mask:
            batch.avail_actions = AgentGroupedTensor(kwargs["avail_actions"], self.agent_grouping)

        if self.use_rnn:
            batch.filled_masks = kwargs["filled_masks"]

        joint_actions = kwargs["joint_actions"]
        epsilon = kwargs["epsilon"]

        with tf.GradientTape(persistent=True) as tape:
            # initial hidden states for rnn
            rnn_states_actor = self.model.init_actor_rnn_states(batch.batch_size)
            rnn_states_critic = self.model.init_critic_rnn_states(batch.batch_size)

            # feedforward
            policy_outputs = self.model(
                observations=batch.observations,
                agent_indices=batch.agent_indices,
                avail_actions=batch.avail_actions,
                rnn_states=rnn_states_actor,
                epsilon=epsilon
            )
            value_outputs = self.model.get_values(
                states=batch.global_states,
                observations=batch.observations,
                joint_actions=joint_actions,
                agent_indices=batch.agent_indices,
                rnn_states=rnn_states_critic,
                target=False
            )
            values_pred = value_outputs.values

            # calculate actor and critic losses
            loss_a, loss_c = [], []
            for group, n_agents in self.n_group_agents.items():
                mask_values = tf.reshape(batch.valid_mask(group, n_agents), [-1])

                dist = policy_outputs.distributions[group]
                pi_probs = dist.probs
                returns = tf.reshape(batch.returns.packed(group), [-1])
                returns = tf.stop_gradient(returns)

                if self.use_actions_mask:  # mask out the unavailable actions.
                    avail_actions = tf.cast(batch.avail_actions.packed(group), tf.bool)
                    pi_probs = tf.where(avail_actions, pi_probs, tf.zeros_like(pi_probs))
                    pi_probs = tf.math.divide_no_nan(pi_probs, tf.reduce_sum(pi_probs, axis=-1, keepdims=True))
                baseline = tf.reshape(tf.reduce_sum(pi_probs * values_pred.packed(group), axis=-1), [-1])

                actions = tf.cast(batch.actions.packed(group), tf.int32)
                action_indices = tf.expand_dims(actions, axis=-1)
                batch_dims = actions.shape.rank
                pi_taken = tf.gather(pi_probs, indices=action_indices, axis=-1, batch_dims=batch_dims)
                q_taken = tf.gather(values_pred.packed(group), indices=action_indices, axis=-1, batch_dims=batch_dims)
                q_taken = tf.reshape(q_taken, [-1])

                log_pi_taken = tf.reshape(tf.math.log(tf.maximum(pi_taken, 1e-8)), [-1])
                advantages = tf.stop_gradient(q_taken - baseline)

                masked_loss_a = tf.reduce_sum(advantages * log_pi_taken * mask_values)
                loss_a.append(-masked_loss_a / tf.reduce_sum(mask_values))

                td_error = (q_taken - returns) * mask_values
                loss_c.append(tf.reduce_sum(td_error ** 2) / tf.reduce_sum(mask_values))

            # update critic
            loss_critic = sum(loss_c)
            gradients_critic = tape.gradient(loss_critic, self.model.critics.trainable_variables)
            if self.use_grad_clip:
                gradients_critic, _ = tf.clip_by_global_norm(gradients_critic, clip_norm=self.grad_clip_norm)
            self.optimizer['critic'].apply_gradients(zip(gradients_critic, self.model.critics.trainable_variables))

            # update actor
            loss_coma = sum(loss_a)
            gradients_actor = tape.gradient(loss_coma, self.model.actors.trainable_variables)
            if self.use_grad_clip:
                gradients_actor, _ = tf.clip_by_global_norm(gradients_actor, clip_norm=self.grad_clip_norm)
            self.optimizer['actor'].apply_gradients(zip(gradients_actor, self.model.actors.trainable_variables))

        return info_train

    def update(self, sample, epsilon=0.0):
        self.iterations += 1

        # prepare training data
        batch = self.build_training_data(sample=sample,
                                         use_actions_mask=self.use_actions_mask,
                                         use_global_state=True)

        joint_actions = tf.concat([one_hot(tf.cast(v, dtype=tf.int32), depth=self.n_actions[k])
                                   for k, v in sample['actions'].items()], axis=-1)
        if self.use_rnn:
            joint_actions = tf.reshape(joint_actions, [batch.batch_size, batch.seq_length, -1])
        else:
            joint_actions = tf.reshape(joint_actions, [batch.batch_size, -1])

        info = self.callback.on_update_start(self.iterations, model=self.model, batch=batch)

        inputs_learn = {
            "batch_size": batch.batch_size,
            "seq_length": batch.seq_length,
            "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float32),
            "global_states": batch.global_states,
            "joint_actions": joint_actions,
            "observations": batch.observations.grouped_tensor,
            "actions": batch.actions.grouped_tensor,
            "returns": batch.returns.grouped_tensor,
            "agent_masks": batch.agent_masks.grouped_tensor,
            "agent_indices": batch.agent_indices.grouped_tensor,
        }

        if self.use_actions_mask:
            inputs_learn["avail_actions"] = batch.avail_actions.grouped_tensor

        if self.use_rnn:
            inputs_learn["filled_masks"] = batch.filled_masks

        info_train = self.learn(**inputs_learn)

        info.update(info_train)

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info))

        return info
