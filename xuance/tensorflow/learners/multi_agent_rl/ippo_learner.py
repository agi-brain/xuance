"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from operator import itemgetter
from xuance.common import List, Optional
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import LearnerMAS
from xuance.tensorflow.utils import ValueNorm


class IPPO_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module):
        super(IPPO_Learner, self).__init__(config, model_keys, agent_keys, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.model_keys}
        else:
            self.value_normalizer = None

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled, IDs = None, None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)
            actions_tensor = np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)
            values_tensor = np.stack(itemgetter(*self.agent_keys)(sample['values']), axis=1)
            returns_tensor = np.stack(itemgetter(*self.agent_keys)(sample['returns']), axis=1)
            advantages_tensor = np.stack(itemgetter(*self.agent_keys)(sample['advantages']), 1)
            log_pi_old_tensor = np.stack(itemgetter(*self.agent_keys)(sample['log_pi_old']), 1)
            ter_tensor = np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1).astype(np.float32)
            msk_tensor = np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1).astype(np.float32)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape([bs, seq_length, -1])}
                if len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape([bs, seq_length])}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: actions_tensor.reshape([bs, seq_length, -1])}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape([bs, seq_length])}
                returns = {k: returns_tensor.reshape([bs, seq_length])}
                advantages = {k: advantages_tensor.reshape([bs, seq_length])}
                log_pi_old = {k: log_pi_old_tensor.reshape([bs, seq_length])}
                terminals = {k: ter_tensor.reshape([bs, seq_length])}
                agent_mask = {k: msk_tensor.reshape([bs, seq_length])}
                IDs = np.eye(self.n_agents, dtype=np.float32)[None, :, None].repeat(
                    batch_size, axis=0).repeat(seq_length, axis=2).reshape([bs, seq_length, self.n_agents])
            else:
                obs = {k: obs_tensor.reshape([bs, -1])}
                if len(actions_tensor.shape) == 2:
                    actions = {k: actions_tensor.reshape(bs)}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape([bs, -1])}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape(bs)}
                returns = {k: returns_tensor.reshape(bs)}
                advantages = {k: advantages_tensor.reshape(bs)}
                log_pi_old = {k: log_pi_old_tensor.reshape(bs)}
                terminals = {k: ter_tensor.reshape(bs)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                IDs = np.eye(self.n_agents, dtype=np.float32)[None].repeat(
                    batch_size, axis=0).reshape(bs, self.n_agents)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: avail_a.reshape([bs, seq_length, -1]).astype(np.float32)}
                else:
                    avail_actions = {k: avail_a.reshape([bs, -1]).astype(np.float32)}

        else:
            obs = {k: sample['obs'][k] for k in self.agent_keys}
            actions = {k: sample['actions'][k] for k in self.agent_keys}
            values = {k: sample['values'][k] for k in self.agent_keys}
            returns = {k: sample['returns'][k] for k in self.agent_keys}
            advantages = {k: sample['advantages'][k] for k in self.agent_keys}
            log_pi_old = {k: sample['log_pi_old'][k] for k in self.agent_keys}
            terminals = {k: sample['terminals'][k].astype(np.float32) for k in self.agent_keys}
            agent_mask = {k: sample['agent_mask'][k].astype(np.float32) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: sample['avail_actions'][k].astype(np.float32) for k in self.agent_keys}

        if use_global_state:
            state = sample['state']

        if self.use_rnn:
            filled = sample['filled'].astype(np.float32)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'obs': obs,
            'actions': actions,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        values = sample_Tensor['values']
        returns = sample_Tensor['returns']
        advantages = sample_Tensor['advantages']
        log_pi_old = sample_Tensor['log_pi_old']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        with tf.GradientTape() as tape:
            # feedforward
            _, _ = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            pi_dists_dict = {key: self.policy.actor[key].dist for key in self.model_keys}
            _, value_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)

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
