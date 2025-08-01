"""
Independent Proximal Policy Optimization (IPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: TensorFlow 2.X
"""
import numpy as np
from argparse import Namespace
from xuance.common import List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.utils import ValueNorm
from xuance.tensorflow.learners.multi_agent_rl.iac_learner import IAC_Learner


class IPPO_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(IPPO_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
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

    # @tf.function
    def forward_fn(self, *args):
        bs, obs, actions, log_pi_old, agent_mask, avail_actions, values, returns, advantages, IDs = args
        with tf.GradientTape() as tape:
            loss_a, loss_e, loss_c = [], [], []
            if self.is_continuous:
                _, pi_mu, pi_std = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
                for key in self.model_keys:
                    mask_values = agent_mask[key]
                    mask_values_sum = tf.reduce_sum(mask_values)
                    log_2pi = tf.math.log(2.0 * np.pi)
                    # policy gradient loss
                    log_std = tf.math.log(pi_std[key] + 1e-8)
                    log_prob = -0.5 * (
                                ((actions[key] - pi_mu[key]) / (pi_std[key] + 1e-8)) ** 2 + 2.0 * log_std + log_2pi)
                    log_pi = tf.reduce_sum(log_prob, axis=-1, keepdims=False)
                    ratio = tf.exp(log_pi - log_pi_old[key])
                    advantages_mask = advantages[key] * mask_values
                    surrogate1 = ratio * advantages_mask
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                    pg_loss = -tf.reduce_sum(tf.minimum(surrogate1, surrogate2)) / mask_values_sum
                    loss_a.append(pg_loss)

                    # entropy loss
                    entropy = tf.reduce_sum(0.5 + 0.5 * log_2pi + log_std, axis=-1, keepdims=True)
                    entropy_loss = tf.reduce_sum(entropy * mask_values) / mask_values_sum
                    loss_e.append(entropy_loss)
            else:
                _, pi_logits = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
                for key in self.model_keys:
                    mask_values = agent_mask[key]
                    mask_values_sum = tf.reduce_sum(mask_values)
                    # policy gradient loss
                    log_prob = tf.nn.log_softmax(pi_logits[key], axis=-1)
                    log_pi = tf.gather(log_prob, actions[key], axis=-1, batch_dims=-1)
                    log_pi = tf.squeeze(log_pi, axis=-1)
                    ratio = tf.exp(log_pi - log_pi_old[key])
                    advantages_mask = advantages[key] * mask_values
                    surrogate1 = ratio * advantages_mask
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                    pg_loss = -tf.reduce_sum(tf.minimum(surrogate1, surrogate2)) / mask_values_sum
                    loss_a.append(pg_loss)

                    # entropy loss
                    probs = tf.exp(log_prob)
                    entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=False)
                    entropy_loss = tf.reduce_sum(entropy * mask_values) / mask_values_sum
                    loss_e.append(entropy_loss)

            _, values_pred_dict = self.policy.get_values(observation=obs, agent_ids=IDs)
            for key in self.model_keys:
                # value loss
                value_pred_i = tf.reshape(values_pred_dict[key], [bs])
                value_target = tf.reshape(returns[key], [bs])
                values_i = tf.reshape(values[key], [bs])
                if self.use_value_clip:
                    value_clipped = values_i + tf.clip_by_value(value_pred_i - values_i,
                                                                -self.value_clip_range, self.value_clip_range)
                    if self.use_value_norm:
                        self.value_normalizer[key].update(tf.reshape(value_target, [bs, 1]))
                        value_target = tf.reshape(self.value_normalizer[key].normalize(tf.reshape(value_target,
                                                                                                  [bs, 1])), [bs])
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i, self.huber_delta)
                        loss_v_clipped = tk.losses.huber(value_target, value_clipped, self.huber_delta)
                    else:
                        loss_v = (value_pred_i - value_target) ** 2
                        loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c_ = tf.maximum(loss_v, loss_v_clipped) * mask_values
                    loss_c.append(tf.reduce_sum(loss_c_) / mask_values_sum)
                else:
                    if self.use_value_norm:
                        self.value_normalizer[key].update(value_target)
                        value_target = self.value_normalizer[key].normalize(value_target)
                    if self.use_huber_loss:
                        loss_v = tk.losses.huber(value_target, value_pred_i, self.huber_delta) * mask_values
                    else:
                        loss_v = ((value_pred_i - value_target) ** 2) * mask_values
                    loss_c.append(tf.reduce_sum(loss_v) / mask_values_sum)

            # Total loss
            loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return loss, loss_a, loss_c, loss_e, values_pred_dict

    # @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            loss, a_loss, c_loss, e_loss, v_pred = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, v_pred, axis=None))
        else:
            return self.forward_fn(*inputs)

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

        loss, a_loss, c_loss, e_loss, v_pred = self.learn(bs, obs, actions, log_pi_old, agent_mask, avail_actions,
                                                          values, returns, advantages, IDs)

        info.update({f"predict_value/{key}": tf.reduce_mean(v_pred[key]).numpy() for key in self.model_keys})

        info.update({
            # "learning_rate": lr,
            "pg_loss": sum(a_loss).numpy(),
            "vf_loss": sum(c_loss).numpy(),
            "entropy_loss": sum(e_loss).numpy(),
            "loss": loss.numpy(),
        })

        return info
