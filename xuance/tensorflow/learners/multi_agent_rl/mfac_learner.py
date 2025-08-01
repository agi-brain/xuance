"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: TensorFlow 2.X
"""
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners.multi_agent_rl.ippo_learner import IPPO_Learner


class MFAC_Learner(IPPO_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        super(MFAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)

    def build_actions_mean_input(self, sample: Optional[dict], use_parameter_sharing: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            if self.n_agents == 1:
                actions_mean_tensor = tf.convert_to_tensor(sample['actions_mean'][k][:, None])
            else:
                actions_mean_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['actions_mean']), axis=1)
            if self.use_rnn:
                actions_mean = {k: tf.reshape(actions_mean_tensor, [bs, seq_length, -1])}
            else:
                actions_mean = {k: tf.reshape(actions_mean_tensor, [bs, -1])}
        else:
            actions_mean = {k: tf.convert_to_tensor(sample['actions_mean'][k]) for k in self.agent_keys}

        return actions_mean

    # @tf.function
    def forward_fn(self, *args):
        bs, obs, actions, act_mean, values, returns, advantages, log_pi_old, agent_mask, avail_actions, IDs = args
        info_train, gradients = {}, {}
        with tf.GradientTape(persistent=True) as tape:
            # feedforward
            _, pi_logits_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            _, values_pred_dict = self.policy.get_values(observation=obs, actions_mean=act_mean, agent_ids=IDs)

            loss_a, loss_e, loss_c = [], [], []
            for key in self.model_keys:
                mask_values = agent_mask[key]
                mask_values_sum = tf.reduce_sum(mask_values)
                # actor loss
                pi_logits = pi_logits_dict[key] / self.policy.temperature
                log_pi = tf.reshape(tf.gather(tf.nn.log_softmax(pi_logits, axis=-1), actions[key],
                                              axis=-1, batch_dims=-1), [bs])
                ratio = tf.exp(log_pi - log_pi_old[key])
                advantages_mask = tf.stop_gradient(advantages[key] * mask_values)
                surrogate1 = ratio * advantages_mask
                surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                pg_loss = -tf.reduce_sum(tf.minimum(surrogate1, surrogate2)) / mask_values_sum
                loss_a.append(pg_loss)

                # entropy loss
                probs = tf.exp(log_pi)
                entropy = -tf.reduce_sum(probs * log_pi, axis=-1, keepdims=False)
                entropy_loss = tf.reduce_sum(entropy * mask_values) / mask_values_sum
                loss_e.append(entropy_loss)

                # critic loss
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

                info_train.update({
                    f"{key}/actor_loss": loss_a[-1],
                    f"{key}/critic_loss": loss_c[-1],
                    f"{key}/entropy": loss_e[-1],
                    f"{key}/predict_value": tf.reduce_mean(value_pred_i),
                })

            loss = sum(loss_a) + self.vf_coef * sum(loss_c) - self.ent_coef * sum(loss_e)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

            info_train.update({
                "loss": loss,
            })

        return info_train

    # @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            info_train = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return info_train[0]
        else:
            return self.forward_fn(*inputs)

    def update(self, sample):
        self.iterations += 1

        # prepare training data
        act_mean = self.build_actions_mean_input(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing)
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

        info = self.callback.on_update_start(self.iterations, method="update", actions_mean=act_mean,
                                             policy=self.policy, sample_Tensor=sample_Tensor, bs=bs)

        info_train = self.learn(bs, obs, actions, act_mean, values, returns, advantages, log_pi_old,
                                agent_mask, avail_actions, IDs)
        for k, v in info_train.items():
            info_train[k] = v.numpy()
        info.update(info_train)

        info.update(self.callback.on_update_end(self.iterations, method="update", policy=self.policy, info=info))

        return info
