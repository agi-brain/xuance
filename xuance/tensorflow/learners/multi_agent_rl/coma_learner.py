"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: TensorFlow 2.X
"""
from tensorflow import one_hot
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.common import List
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

    # @tf.function
    def forward_fn(self, *args):
        bs, batch_size, obs, state, actions, agent_mask, avail_actions, returns, IDs, epsilon = args
        with tf.GradientTape(persistent=True) as tape:
            _, pi_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)

            if self.use_parameter_sharing:
                key = self.model_keys[0]
                actions_onehot = {key: one_hot(actions[key], self.n_actions[key])}
            else:
                IDs = tf.reshape(tf.tile(tf.eye(self.n_agents)[None], [batch_size, 1, 1]), [bs, -1])
                actions_onehot = {k: one_hot(actions[k], self.n_actions[k]) for k in self.agent_keys}

            _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot, agent_ids=IDs)

            if self.use_parameter_sharing:
                values_pred_dict = {k: tf.reshape(values_pred, [bs, -1]) for k in self.model_keys}
            else:
                values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

            loss_a, loss_c = [], []
            for key in self.model_keys:
                mask_values = agent_mask[key]
                mask_values_sum = tf.reduce_sum(mask_values)

                pi_probs = tf.nn.softmax(pi_logits[key], axis=-1)
                pi_probs = (1 - epsilon) * pi_probs + epsilon * 1 / self.n_actions[key]
                baseline = tf.reshape(tf.reduce_sum(pi_probs * values_pred_dict[key], -1), [bs])
                pi_taken = tf.gather(pi_probs, actions[key], axis=-1, batch_dims=-1)
                q_taken = tf.reshape(tf.gather(values_pred_dict[key], actions[key], axis=-1, batch_dims=-1), [bs])
                log_pi_taken = tf.reshape(tf.math.log(pi_taken), [bs])
                advantages = tf.stop_gradient(q_taken - baseline)
                loss_a.append(-tf.reduce_sum(advantages * log_pi_taken * mask_values) / mask_values_sum)

                td_error = (q_taken - returns[key]) * mask_values
                loss_c.append(tf.reduce_sum(td_error ** 2) / mask_values_sum)

            # update critic
            loss_critic = sum(loss_c)
            gradients_critic = tape.gradient(loss_critic, self.policy.parameters_critic)
            if self.use_grad_clip:
                gradients_critic, _ = tf.clip_by_global_norm(gradients_critic, clip_norm=self.grad_clip_norm)
                self.optimizer['critic'].apply_gradients(zip(gradients_critic, self.policy.parameters_critic))
            else:
                self.optimizer['critic'].apply_gradients(zip(gradients_critic, self.policy.parameters_critic))

            # update actor
            loss_coma = sum(loss_a)
            gradients_actor = tape.gradient(loss_coma, self.policy.parameters_actor)
            if self.use_grad_clip:
                gradients_actor, _ = tf.clip_by_global_norm(gradients_actor, clip_norm=self.grad_clip_norm)
                self.optimizer['actor'].apply_gradients(zip(gradients_actor, self.policy.parameters_actor))
            else:
                self.optimizer['actor'].apply_gradients(zip(gradients_actor, self.policy.parameters_actor))

        return loss_coma, loss_critic

    # @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss_coma, loss_critic = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_coma, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_critic, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, sample, epsilon=0.0):
        self.iterations += 1

        # prepare training data
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

        loss_coma, loss_critic = self.learn(bs, batch_size, obs, state, actions,
                                            agent_mask, avail_actions, returns, IDs, epsilon)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update({
            "actor_loss": loss_coma.numpy(),
            "critic_loss": loss_critic.numpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
