"""
Proximal Policy Optimization with KL divergence (PPO-KL)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner
from xuance.tensorflow.utils import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPOKL_Learner, self).__init__(config, policy)
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

    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        with tf.GradientTape() as tape:
            outputs, _, v_pred = self.policy(obs_batch)
            a_dist = self.policy.actor.dist
            log_prob = a_dist.log_prob(act_batch)
            old_dist = merge_distributions(old_dists)
            kl = tf.reduce_mean(a_dist.kl_divergence(old_dist))
            old_logp_batch = old_dist.log_prob(act_batch)

            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob - old_logp_batch)
            a_loss = -tf.reduce_mean(ratio * adv_batch) + self.kl_coef * kl
            c_loss = tk.losses.mean_squared_error(ret_batch, v_pred)
            e_loss = tf.reduce_mean(a_dist.entropy())
            loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
            if kl > self.target_kl * 1.5:
                self.kl_coef = self.kl_coef * 2.
            elif kl < self.target_kl * 0.5:
                self.kl_coef = self.kl_coef / 2.
            self.kl_coef = tf.clip_by_value(self.kl_coef, 0.1, 20)
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
        obs_batch = samples['obs']
        act_batch = samples['actions']
        ret_batch = samples['returns']
        adv_batch = samples['advantages']
        old_dists = samples['aux_batch']['old_dist']

        a_loss, c_loss, e_loss, kl, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch, old_dists)

        info = {
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "kl": kl.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy()
        }

        return info
