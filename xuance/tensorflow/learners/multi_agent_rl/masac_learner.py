"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, Module
from xuance.tensorflow.learners.multi_agent_rl.isac_learner import ISAC_Learner


class MASAC_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MASAC_Learner, self).__init__(config, model_keys, agent_keys, policy)

    def update(self, sample):
        self.iterations += 1
        info = {}

        # Prepare training data.
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_joint = obs[key].reshape(batch_size, -1)
            next_obs_joint = obs_next[key].reshape(batch_size, -1)
            actions_joint = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            obs_joint = np.stack(itemgetter(*self.agent_keys)(obs), axis=-1).reshape(batch_size, -1)
            next_obs_joint = np.stack(itemgetter(*self.agent_keys)(obs_next), axis=-1).reshape(batch_size, -1)
            actions_joint = np.stack(itemgetter(*self.agent_keys)(actions), axis=-1).reshape(batch_size, -1)

        # Update critic
        _, actions_next, log_pi_next = self.policy(observation=obs_next, agent_ids=IDs)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = tf.reshape(tf.reshape(actions_next[key], [batch_size, self.n_agents, -1]),
                                            [batch_size, -1])
        else:
            actions_next_joint = tf.reshape(tf.concat(itemgetter(*self.model_keys)(actions_next), -1),
                                            [batch_size, -1])
        _, _, action_q_1, action_q_2 = self.policy.Qaction(joint_observation=obs_joint, joint_actions=actions_joint,
                                                           agent_ids=IDs)
        _, _, target_q = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint,
                                             agent_ids=IDs)
        for key in self.model_keys:
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                action_q_1_i = tf.reshape(action_q_1[key], [bs])
                action_q_2_i = tf.reshape(action_q_2[key], [bs])
                log_pi_next_eval = tf.reshape(log_pi_next[key], [bs])
                target_value = tf.reshape(target_q[key], [bs]) - self.alpha[key] * log_pi_next_eval
                backup = rewards[key] + (1 - terminals[key]) * self.gamma * target_value
                backup = tf.stop_gradient(backup)
                td_error_1, td_error_2 = action_q_1_i - backup, action_q_2_i - backup
                td_error_1 *= mask_values
                td_error_2 *= mask_values
                loss_c = (tf.reduce_sum(td_error_1 ** 2) + tf.reduce_sum(td_error_2 ** 2)) / tf.reduce_sum(mask_values)
                gradients = tape.gradient(loss_c, self.policy.critic_trainable_variables(key))
                if self.use_grad_clip:
                    self.optimizer[key]['critic'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.critic_trainable_variables(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer[key]['critic'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.critic_trainable_variables(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_critic": loss_c.numpy()})

        # Update actor
        _, actions_eval, log_pi_eval = self.policy(observation=obs, agent_ids=IDs)
        log_pi_eval_i = {}
        for key in self.model_keys:
            with tf.GradientTape() as tape:
                mask_values = agent_mask[key]
                if self.use_parameter_sharing:
                    actions_eval_joint = tf.reshape(tf.reshape(actions_eval[key], [batch_size, self.n_agents, -1]),
                                                    [batch_size, -1])
                else:
                    actions_eval_detach_others = {k: actions_eval[k] if k == key else tf.stop_gradient(actions_eval[k])
                                                  for k in self.model_keys}
                    actions_eval_joint = tf.reshape(tf.concat(itemgetter(*self.model_keys)(actions_eval_detach_others),
                                                              axis=-1), [batch_size, -1])
                _, _, policy_q_1, policy_q_2 = self.policy.Qpolicy(joint_observation=obs_joint,
                                                                   joint_actions=actions_eval_joint,
                                                                   agent_ids=IDs, agent_key=key)
                log_pi_eval_i[key] = tf.reshape(log_pi_eval[key], [bs])
                policy_q = tf.reshape(tf.math.minimum(policy_q_1[key], policy_q_2[key]), [bs])
                loss_a = tf.reduce_sum((self.alpha[key] * log_pi_eval_i[key] - policy_q) * mask_values) / tf.reduce_sum(
                    mask_values)
                gradients = tape.gradient(loss_a, self.policy.actor_trainable_variables(key))
                if self.use_grad_clip:
                    self.optimizer[key]['actor'].apply_gradients([
                        (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                        for (grad, var) in zip(gradients, self.policy.actor_trainable_variables(key))
                        if grad is not None
                    ])
                else:
                    self.optimizer[key]['actor'].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.policy.actor_trainable_variables(key))
                        if grad is not None
                    ])

                info.update({f"{key}/loss_actor": loss_a.numpy(),
                             f"{key}/predictQ": tf.math.reduce_mean(policy_q).numpy()})

        # Automatically entropy tuning
        if self.use_automatic_entropy_tuning:
            for key in self.model_keys:
                with tf.GradientTape() as tape:
                    alpha_loss = -tf.math.reduce_mean(
                        self.alpha_layer[key].log_alpha.value() * (log_pi_eval_i[key] + self.target_entropy[key]))
                    gradients = tape.gradient(alpha_loss, self.alpha_layer[key].trainable_variables)
                    self.alpha_optimizer[key].apply_gradients([
                        (grad, var)
                        for (grad, var) in zip(gradients, self.alpha_layer[key].trainable_variables)
                        if grad is not None
                    ])
                    self.alpha[key] = tf.math.exp(self.alpha_layer[key].log_alpha)
                    info.update({f"{key}/alpha_loss": alpha_loss.numpy(),
                                 f"{key}/alpha": self.alpha[key].numpy()})
        else:
            for key in self.model_keys:
                info.update({f"{key}/alpha_loss": 0.0,
                             f"{key}/alpha": self.alpha[key]})

        self.policy.soft_update(self.tau)
        return info
