"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow 2.X
"""
from tensorflow import one_hot
from xuance.tensorflow import tf, tk, Module
from xuance.common import List
from argparse import Namespace
from xuance.tensorflow.learners.multi_agent_rl.iac_learner import IAC_Learner


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = {
                'actor': tk.optimizers.legacy.Adam(self.config.learning_rate_actor),
                'critic': tk.optimizers.legacy.Adam(self.config.learning_rate_critic),
            }
        else:
            self.optimizer = {
                'actor': tk.optimizers.Adam(self.config.learning_rate_actor),
                'critic': tk.optimizers.Adam(self.config.learning_rate_critic)
            }

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        returns = sample_Tensor['returns']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        info = self.callback.on_update_start(self.iterations, method="update",
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: one_hot(actions[key], self.n_actions[key])}
        else:
            IDs = tf.reshape(tf.tile(tf.expand_dims(tf.eye(self.n_agents), 0), [batch_size, 1, 1]), [bs, -1])
            actions_onehot = {k: one_hot(actions[k], self.n_actions[k]) for k in self.agent_keys}

        # update critic
        with tf.GradientTape() as tape:
            _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                    agent_ids=IDs, target=False)

            if self.use_parameter_sharing:
                values_pred_dict = {k: tf.reshape(values_pred, [bs, -1]) for k in self.model_keys}
            else:
                values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

            # calculate loss
            loss_c = []
            for key in self.model_keys:
                mask_values = agent_mask[key]

                action_indices = tf.cast(actions[key], dtype=tf.int32)
                q_taken = tf.reshape(tf.gather(values_pred_dict[key], tf.expand_dims(action_indices, axis=1),
                                               batch_dims=-1), [bs])

                td_error = (q_taken - tf.stop_gradient(returns[key])) * mask_values
                loss_c.append(tf.reduce_sum((td_error ** 2)) / tf.reduce_sum(mask_values))

                info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_critic",
                                                               mask_values=mask_values, td_error=td_error))

            # update critic
            loss_critic = sum(loss_c)
            gradients = tape.gradient(loss_critic, self.policy.parameters_critic)
            if self.use_grad_clip:
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])
            else:
                self.optimizer['critic'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.parameters_critic)
                    if grad is not None
                ])
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

        # update actor(s)
        with tf.GradientTape() as tape:
            # feedforward
            _, pi_probs = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=epsilon)

            loss_a = []
            for key in self.model_keys:
                mask_values = agent_mask[key]
                if self.use_actions_mask:
                    pi_probs[key][avail_actions[key] == 0] = 0.0  # mask out the unavailable actions.
                    pi_probs[key] = pi_probs[key] / pi_probs[key].sum(dim=-1, keepdim=True)  # re-normalize the actions.
                    pi_probs[key][avail_actions[key] == 0] = 0.0
                baseline = tf.reshape(tf.reduce_sum(pi_probs[key] * values_pred_dict[key], axis=-1), [bs])
                action_indices = tf.cast(actions[key], dtype=tf.int32)
                pi_taken = tf.gather(pi_probs[key], tf.expand_dims(action_indices, axis=-1), batch_dims=-1)
                q_taken = tf.reshape(tf.gather(values_pred_dict[key], tf.expand_dims(action_indices, axis=1),
                                               batch_dims=-1), [bs])
                log_pi_taken = tf.reshape(tf.math.log(pi_taken), [bs])
                advantages = tf.stop_gradient(q_taken - baseline)
                loss_a.append(-tf.reduce_sum(advantages * log_pi_taken * mask_values) / tf.reduce_sum(mask_values))

                info.update(self.callback.on_update_agent_wise(self.iterations, key, info=info, method="update_actor",
                                                               mask_values=mask_values, pi_probs=pi_probs,
                                                               baseline=baseline, pi_taken=pi_taken,
                                                               q_taken=q_taken, log_pi_taken=log_pi_taken,
                                                               advantages=advantages, loss_a=loss_a))

            loss_coma = sum(loss_a)
            gradients = tape.gradient(loss_coma, self.policy.parameters_actor)
            if self.use_grad_clip:
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])
            else:
                self.optimizer['actor'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.parameters_actor)
                    if grad is not None
                ])
            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

        # Logger
        learning_rate_actor = self.optimizer['actor']._decayed_lr(tf.float32)
        learning_rate_critic = self.optimizer['critic']._decayed_lr(tf.float32)

        info.update({
            "learning_rate_actor": learning_rate_actor.numpy(),
            "learning_rate_critic": learning_rate_critic.numpy(),
            "actor_loss": loss_coma.numpy(),
            "critic_loss": loss_critic.numpy(),
            "advantage": tf.math.reduce_mean(advantages).numpy()
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info

    def update_rnn(self, sample, epsilon=0.0):
        self.iterations += 1

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)

        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        returns = sample_Tensor['returns']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = tf.reshape(tf.tile(tf.expand_dims(filled, axis=1),
                                        [batch_size, self.n_agents, seq_len]), [bs_rnn, seq_len])
        else:
            IDs = tf.tile(tf.eye(self.n_agents)[None, None, :, :], [batch_size, seq_len, 1, 1])

        info = self.callback.on_update_start(self.iterations, method="update_rnn",
                                             policy=self.policy, sample_Tensor=sample_Tensor,
                                             bs_rnn=bs_rnn, filled=filled, IDs=IDs)

        return info