"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link:
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS


class QMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(QMIX_Learner, self).__init__(config, model_keys, agent_keys, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

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
            rewards_tot = rewards[key].mean(axis=1).reshape(batch_size, 1)
            terminals_tot = terminals[key].all(axis=1, keepdims=False).astype(np.float32).reshape(batch_size, 1)
        else:
            bs = batch_size
            rewards_tot = np.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(axis=-1, keepdims=True)
            terminals_tot = np.stack(itemgetter(*self.agent_keys)(terminals),
                                     axis=1).all(axis=1, keepdims=True).astype(np.float32)

        with tf.GradientTape() as tape:
            _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

            q_eval_a, q_next_a = {}, {}
            for key in self.model_keys:
                q_eval_a[key] = tf.reshape(tf.gather(q_eval[key], tf.cast(actions[key][:, None], dtype=tf.int32),
                                                     axis=-1, batch_dims=-1), [bs])
                if self.use_actions_mask:
                    q_next[key][avail_actions_next[key] == 0] = -9999999

                if self.config.double_q:
                    _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                 avail_actions=avail_actions, agent_key=key)
                    q_next_a[key] = tf.reshape(tf.gather(q_next[key], act_next[key][:, None],
                                                         axis=-1, batch_dims=-1), [bs])
                else:
                    q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs)

                q_eval_a[key] *= agent_mask[key]
                q_next_a[key] *= agent_mask[key]

            q_tot_eval = self.policy.Q_tot(q_eval_a, state)
            q_tot_next = self.policy.Qtarget_tot(q_next_a, state_next)
            q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

            q_tot_target = tf.reshape(q_tot_target, [-1])
            q_tot_eval = tf.reshape(q_tot_eval, [-1])

            # calculate the loss function
            loss = tk.losses.mean_squared_error(tf.stop_gradient(q_tot_target), q_tot_eval)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.policy.trainable_variables)
                if grad is not None
            ])

            info.update({
                "loss_Q": loss.numpy(),
                "predictQ": tf.math.reduce_mean(q_tot_eval).numpy()
            })

            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()
        return info
