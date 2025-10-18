"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner
from xuance.tensorflow.utils import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(PPOKL_Learner, self).__init__(config, policy, callback)
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
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.target_kl = config.target_kl
        self.kl_coef = config.kl_coef
        self.mse_loss = tk.losses.MeanSquaredError()
        self.is_continuous = self.policy.is_continuous

    @tf.function
    def forward_fn(self, *args):
        with tf.GradientTape() as tape:
            if self.is_continuous:
                obs_batch, act_batch, ret_batch, adv_batch, old_mu, old_std = args
                outputs, mu, std, v_pred = self.policy(obs_batch)
                # calculate log prob
                log_2pi = tf.math.log(2.0 * np.pi)
                log_std, old_log_std = tf.math.log(std + 1e-8), tf.math.log(old_std + 1e-8)
                log_prob = -0.5 * (((act_batch - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + log_2pi)
                old_log_prob = -0.5 * (((act_batch - old_mu) / (old_std + 1e-8)) ** 2 + 2.0 * old_log_std + log_2pi)
                log_prob_a = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
                old_log_prob_a = tf.reduce_sum(old_log_prob, axis=-1, keepdims=True)
                # calculate entropy
                entropy = tf.reduce_sum(0.5 + 0.5 * tf.math.log(2.0 * np.pi) + log_std, axis=-1, keepdims=True)
                # calculate kl divergence
                var1, var2 = tf.square(std), tf.square(old_std)
                kl = tf.math.log(old_std / std) + (var1 + tf.square(mu - old_mu)) / (2.0 * var2) - 0.5
            else:
                obs_batch, act_batch, ret_batch, adv_batch, old_logits = args
                outputs, logits, v_pred = self.policy(obs_batch)
                # calculate log prob
                log_prob = tf.nn.log_softmax(logits, axis=-1)
                log_prob_a = tf.gather(log_prob, act_batch, axis=-1, batch_dims=-1)
                # calculate entropy
                probs = tf.exp(log_prob)
                entropy = -tf.reduce_sum(probs * log_prob, axis=-1, keepdims=True)
                # calculate kl divergence
                old_log_prob_batch = tf.nn.log_softmax(old_logits, axis=-1)  # log Q(a)
                old_log_prob_a = tf.gather(old_log_prob_batch, act_batch, axis=-1, batch_dims=-1)
                kl = tf.reduce_sum(probs * (log_prob - old_log_prob_batch), axis=-1, keepdims=True)

            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob_a - old_log_prob_a)
            kl = tf.reduce_mean(kl)
            a_loss = -tf.reduce_mean(ratio * adv_batch) + self.kl_coef * kl
            c_loss = self.mse_loss(ret_batch, v_pred)
            e_loss = tf.reduce_mean(entropy)

            loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return a_loss, c_loss, e_loss, kl, v_pred

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, c_loss, e_loss, kl, v_pred = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, kl, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, v_pred, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'][:, None], dtype=tf.float32)
        if self.is_continuous:
            act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        else:
            act_batch = tf.convert_to_tensor(samples["actions"][:, None], dtype=tf.int32)
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch, old_dists=old_dists)
        if self.is_continuous:
            old_mu = old_dists.mu
            old_std = old_dists.std
            a_loss, c_loss, e_loss, kl, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch, old_mu, old_std)
        else:
            old_logits = old_dists.logits
            a_loss, c_loss, e_loss, kl, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch, old_logits)

        if kl > self.target_kl * 1.5:
            self.kl_coef = self.kl_coef * 2.
        elif kl < self.target_kl * 0.5:
            self.kl_coef = self.kl_coef / 2.
        self.kl_coef = tf.clip_by_value(self.kl_coef, 0.1, 20)

        info.update({
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "kl": kl.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy()
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                v_pred=v_pred, kl=kl, a_loss=a_loss, c_loss=c_loss, e_loss=e_loss))
        return info
