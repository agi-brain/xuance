"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class WQMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(WQMIX_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
        self.build_optimizer()
        self.alpha = config.alpha
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
    def forward_fn(self, bs, batch_size, state, obs, actions, rewards_tot, state_next, obs_next, terminals_tot,
                   agent_mask, avail_actions, avail_actions_next, IDs):
        with tf.GradientTape() as tape:
            # calculate Q_tot
            _, action_max, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            _, q_eval_centralized = self.policy.q_centralized(observation=obs, agent_ids=IDs)
            _, q_eval_next_centralized = self.policy.target_q_centralized(observation=obs_next, agent_ids=IDs)

            q_eval_a, q_eval_centralized_a, q_eval_next_centralized_a, act_next = {}, {}, {}, {}
            for key in self.model_keys:
                actions_eval = tf.cast(actions[key][:, None], dtype=tf.int32)
                q_eval_a[key] = tf.reshape(tf.gather(q_eval[key], actions_eval, axis=-1, batch_dims=-1), [bs])
                q_eval_centralized_a[key] = tf.reshape(tf.gather(q_eval_centralized[key], actions_eval,
                                                                 axis=-1, batch_dims=-1), [bs])

                if self.config.double_q:
                    _, a_next_greedy, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                      avail_actions=avail_actions_next, agent_key=key)
                    act_next[key] = tf.expand_dims(a_next_greedy[key], axis=-1)
                else:
                    _, q_next_eval = self.policy.Qtarget(observation=obs_next, agent_ids=IDs, agent_key=key)
                    if self.use_actions_mask:
                        q_next_eval[key][avail_actions_next[key] == 0] = -1e10
                    act_next[key] = tf.argmax(q_next_eval[key], axis=-1)
                q_eval_next_centralized_a[key] = tf.reshape(tf.gather(q_eval_next_centralized[key], act_next[key],
                                                                      axis=-1, batch_dims=-1), [bs])

                q_eval_a[key] *= agent_mask[key]
                q_eval_centralized_a[key] *= agent_mask[key]
                q_eval_next_centralized_a[key] *= agent_mask[key]

            q_tot_eval = self.policy.Q_tot(q_eval_a, state)  # calculate Q_tot
            q_tot_centralized = self.policy.q_feedforward(q_eval_centralized_a, state)  # calculate centralized Q
            q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized_a, state_next)  # y_i

            target_value = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next_centralized
            target_value = tf.stop_gradient(target_value)
            td_error = q_tot_eval - target_value

            # calculate weights
            ones = tf.ones_like(td_error)
            w = ones * self.alpha
            if self.config.agent == "CWQMIX":
                condition_1 = ((action_max == actions.reshape([-1, self.n_agents, 1])) * agent_mask).all(dim=1)
                condition_2 = target_value > q_tot_centralized
                conditions = condition_1 | condition_2
                w = tf.where(conditions, ones, w)
            elif self.config.agent == "OWQMIX":
                condition = td_error < 0
                w = tf.where(condition, ones, w)
            else:
                raise AttributeError(f"The agent named is {self.config.agent} is currently not supported.")

            # calculate losses and train
            target_value = tf.reshape(target_value, [batch_size])
            q_tot_centralized = tf.reshape(q_tot_centralized, [batch_size])
            td_error = tf.reshape(td_error, [batch_size])
            w = tf.reshape(w, [batch_size])

            loss_central = self.mse_loss(target_value, q_tot_centralized)
            loss_qmix = tf.reduce_mean(tf.stop_gradient(w) * (td_error ** 2))
            loss = loss_qmix + loss_central

            gradients = tape.gradient(loss, self.policy.parameters_model)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.parameters_model))

        return loss_qmix, loss_central, loss, tf.math.reduce_mean(q_tot_eval)

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss_qmix, loss_central, loss, predictQ = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_qmix, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_central, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None))
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

        info = self.callback.on_update_start(self.iterations, method="update", policy=self.policy,
                                             sample_Tensor=sample_Tensor, bs=bs,
                                             rewards_tot=rewards_tot, terminals_tot=terminals_tot)

        loss_qmix, loss_central, loss, predictQ = self.learn(bs, batch_size, state, obs, actions, rewards_tot,
                                                             state_next, obs_next, terminals_tot,
                                                             agent_mask, avail_actions, avail_actions_next, IDs)

        info.update({
            "loss_Qmix": loss_qmix.numpy(),
            "loss_central": loss_central.numpy(),
            "loss": loss.numpy(),
            "predictQ": predictQ.numpy()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
