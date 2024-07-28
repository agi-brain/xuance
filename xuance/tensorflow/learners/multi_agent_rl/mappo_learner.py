"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners.multi_agent_rl.ippo_learner import IPPO_Learner


class MAPPO_Clip_Learner(IPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(MAPPO_Clip_Learner, self).__init__(config, model_keys, agent_keys, policy)

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=self.use_global_state)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        # prepare critic inputs
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.use_global_state:
                critic_input = {key: state.reshape(batch_size, 1, -1).repeat(self.n_agents, axis=1).reshape(bs, -1)}
            else:
                critic_input = {key: obs[key].reshape(batch_size, 1, -1).repeat(self.n_agents, axis=1).reshape(bs, -1)}
        else:
            bs = batch_size
            if self.use_global_state:
                critic_input = {k: state.reshape(batch_size, -1) for k in self.agent_keys}
            else:
                joint_obs = np.stack(itemgetter(*self.agent_keys)(obs), axis=-1)
                critic_input = {k: joint_obs for k in self.agent_keys}

        with tf.GradientTape() as tape:
            _, pi_dists_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            pi_dists_dict = {key: self.policy.actor[key].dist for key in self.model_keys}
            _, value_pred_dict = self.policy.get_values(observation=critic_input, agent_ids=IDs)

            # calculate losses for each agent
            loss_a, loss_e, loss_c = [], [], []
            for key in self.model_keys:
                mask_values = agent_mask[key]
                # actor loss
                log_pi = tf.reshape(pi_dists_dict[key].log_prob(actions[key]), [bs])
                ratio = tf.reshape(tf.exp(log_pi - log_pi_old[key]), [bs])
                advantages_mask = tf.stop_gradient(advantages[key]) * mask_values
                surrogate1 = ratio * advantages_mask
                surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                loss_a.append(-tf.reduce_mean(tf.math.minimum(surrogate1, surrogate2)))

                # entropy loss
                entropy = tf.reshape(pi_dists_dict[key].entropy(), [bs]) * mask_values
                loss_e.append(tf.reduce_mean(entropy))

                # critic loss
                value_pred_i = tf.reshape(value_pred_dict[key], [bs])
                value_target = tf.reshape(returns[key], [bs])
                values_i = tf.reshape(values[key], [bs])
                if self.use_value_clip:
                    value_clipped = values_i + tf.clip_by_value(value_pred_i - values_i,
                                                                -self.value_clip_range, self.value_clip_range)
                    if self.use_value_norm:
                        self.value_normalizer[key].update(tf.reshape(value_target, [bs, 1]))
                        value_target = self.value_normalizer[key].normalize(tf.reshape(value_target, [bs, 1]))
                        value_target = tf.reshape(value_target, [bs])
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i)
                        loss_v_clipped = tk.losses.huber(value_target, value_clipped)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c_ = tf.math.maximum(loss_v, loss_v_clipped) * mask_values
                    loss_c.append(tf.reduce_sum(loss_c_) / tf.reduce_sum(mask_values))
                else:
                    if self.use_value_norm:
                        self.value_normalizer[key].update(value_target)
                        value_target = self.value_normalizer[key].normalize(value_target)
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i) * mask_values
                    else:
                        loss_v = ((value_pred_i - value_target) ** 2) * mask_values
                    loss_c.append(tf.reduce_sum(loss_v) / tf.reduce_sum(mask_values))

                info.update({f"{key}/actor_loss": loss_a[-1].numpy(),
                             f"{key}/critic_loss": loss_c[-1].numpy(),
                             f"{key}/entropy": loss_e[-1].numpy(),
                             f"{key}/predict_value": tf.reduce_mean(value_pred_i).numpy()})

            loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)

            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            else:
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            info.update({"loss": loss.numpy()})

        return info
