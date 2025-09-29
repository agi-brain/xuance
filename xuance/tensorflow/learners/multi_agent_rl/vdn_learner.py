"""
Value Decomposition Networks (VDN)
Paper link:
https://arxiv.org/pdf/1706.05296.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class VDN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(VDN_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.mse_loss = tk.losses.MeanSquaredError()

    def build_optimizer(self):
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = tk.optimizers.legacy.Adam(self.config.learning_rate)
        else:
            self.optimizer = tk.optimizers.Adam(self.config.learning_rate)

    @tf.function
    def forward_fn(self, bs, obs, actions, rewards_tot, obs_next, terminals_tot,
                   agent_mask, avail_actions, avail_actions_next, IDs):
        with tf.GradientTape() as tape:
            _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

            q_eval_a, q_next_a = {}, {}
            for key in self.model_keys:
                q_eval_a[key] = tf.reshape(tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                                     axis=-1, batch_dims=-1), [bs])
                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -1e10

                if self.config.double_q:
                    _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                 avail_actions=avail_actions, agent_key=key)
                    q_next_a[key] = tf.reshape(tf.gather(q_next[key], act_next[key][:, None],
                                                         axis=-1, batch_dims=-1), [bs])
                else:
                    q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

                q_eval_a[key] *= agent_mask[key]
                q_next_a[key] *= agent_mask[key]

            q_tot_eval = self.policy.Q_tot(q_eval_a)
            q_tot_next = self.policy.Qtarget_tot(q_next_a)
            q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

            q_tot_target = tf.reshape(q_tot_target, [-1])
            q_tot_eval = tf.reshape(q_tot_eval, [-1])

            # calculate the loss function
            loss = self.mse_loss(tf.stop_gradient(q_tot_target), q_tot_eval)

        gradients = tape.gradient(loss, self.policy.parameters_model)
        if self.use_grad_clip:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))
        else:
            self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))

        return loss, tf.math.reduce_mean(q_tot_eval)

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss, predictQ = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None))
        else:
            return self.forward_fn(*inputs)


    def update(self, sample):
        self.iterations += 1

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
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

        loss, predictQ = self.learn(bs, obs, actions, rewards_tot, obs_next, terminals_tot,
                                    agent_mask, avail_actions, avail_actions_next, IDs)
        info.update({
            "loss_Q": loss.numpy(),
            "predictQ": predictQ.numpy()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info,
                                                predictQ=predictQ, loss=loss))

        return info
