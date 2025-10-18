"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner
from xuance.tensorflow.utils import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(PPG_Learner, self).__init__(config, policy, callback)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0
        self.mse_loss = tk.losses.MeanSquaredError()
        self.is_continuous = self.policy.is_continuous

    @tf.function
    def policy_forward_fn(self, obs_batch, act_batch, adv_batch, old_log_prob_batch):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                _, mu, std, _, _ = self.policy(obs_batch)
                # calculate log prob
                log_std = tf.math.log(std + 1e-8)
                log_prob = -0.5 * (((act_batch - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
                log_prob_a = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
                # calculate entropy
                entropy = tf.reduce_sum(0.5 + 0.5 * tf.math.log(2.0 * np.pi) + log_std, axis=-1, keepdims=True)
            else:
                _, logits, _, _ = self.policy(obs_batch)
                # calculate log prob
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                log_prob_a = tf.gather(log_prob, act_batch, axis=-1, batch_dims=-1)
                # calculate entropy
                probs = tf.exp(log_prob)
                entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=True)

            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob_a - old_log_prob_batch)
            surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            surrogate2 = adv_batch * ratio

            a_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            e_loss = tf.reduce_mean(entropy)

            loss = a_loss - self.ent_coef * e_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return a_loss, e_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, ret_batch):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                _, _, _, v_pred, _ = self.policy(obs_batch)
            else:
                _, _, v_pred, _ = self.policy(obs_batch)

            loss = self.mse_loss(ret_batch, v_pred)
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss

    @tf.function
    def auxiliary_forward_fn(self, *args):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                obs_batch, ret_batch, old_mu, old_std = args
                _, mu, std, v, aux_v = self.policy(obs_batch)
                # calculate kl divergence
                var1, var2 = tf.square(std), tf.square(old_std)
                kl = tf.math.log(old_std / std) + (var1 + tf.square(mu - old_mu)) / (2.0 * var2) - 0.5
            else:
                obs_batch, ret_batch, old_logits = args
                _, logits, v, aux_v = self.policy(obs_batch)
                # calculate kl divergence
                log_p = tf.nn.log_softmax(logits, axis=-1)  # log P(a)
                log_q = tf.nn.log_softmax(old_logits, axis=-1)  # log Q(a)
                p = tf.math.exp(log_p)  # P(a)
                kl = tf.reduce_sum(p * (log_p - log_q), axis=-1)

            aux_loss = self.mse_loss(tf.stop_gradient(v), aux_v)
            kl_loss = tf.reduce_mean(kl)
            value_loss = self.mse_loss(ret_batch, v)

            loss = aux_loss + self.kl_beta * kl_loss + value_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss

    @tf.function
    def learn_policy(self, *inputs):
        if self.distributed_training:
            a_loss, e_loss = self.policy.mirrored_strategy.run(self.policy_forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None))
        else:
            return self.policy_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            loss = self.policy.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        else:
            return self.critic_forward_fn(*inputs)

    @tf.function
    def learn_auxiliary(self, *inputs):
        if self.distributed_training:
            loss = self.policy.mirrored_strategy.run(self.auxiliary_forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        else:
            return self.auxiliary_forward_fn(*inputs)

    def update_policy(self, **samples):
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'][:, None], dtype=tf.float32)
        if self.is_continuous:
            act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        else:
            act_batch = tf.convert_to_tensor(samples["actions"][:, None], dtype=tf.int32)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_log_prob_batch = tf.stop_gradient(old_dist.log_prob(act_batch))
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch, advantages=adv_batch,
                                             old_dist=old_dist, old_logp=old_log_prob_batch)

        a_loss, e_loss = self.learn_policy(obs_batch, act_batch, adv_batch, old_log_prob_batch)
        info.update({"actor-loss": a_loss.numpy(), "entropy": e_loss.numpy()})
        self.policy_iterations += 1
        info.update(self.callback.on_update_end(self.iterations, method="update_policy",
                                                policy=self.policy, info=info,
                                                a_loss=a_loss, e_loss=e_loss))
        return info

    def update_critic(self, **samples):
        self.value_iterations += 1
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        info = self.callback.on_update_start(self.iterations, policy=self.policy, obs=obs_batch, returns=ret_batch)

        loss = self.learn_critic(obs_batch, ret_batch)
        info.update({"critic-loss": loss.numpy()})
        info.update(self.callback.on_update_end(self.iterations, method="update_critic",
                                                policy=self.policy, info=info, loss=loss))
        return info

    def update_auxiliary(self, **samples):
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, returns=ret_batch, old_dist=old_dists)
        if self.is_continuous:
            old_mu = old_dists.mu
            old_std = old_dists.std
            loss = self.learn_auxiliary(obs_batch, ret_batch, old_mu, old_std)
        else:
            old_logits = old_dists.logits
            loss = self.learn_auxiliary(obs_batch, ret_batch, old_logits)
        info.update({"kl-loss": loss.numpy()})
        info.update(self.callback.on_update_end(self.iterations, method="update_auxiliary",
                                                policy=self.policy, info=info, loss=loss))
        return info

    def update(self):
        pass
