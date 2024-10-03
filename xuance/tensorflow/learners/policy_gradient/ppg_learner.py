"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner
from xuance.tensorflow.utils import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PPG_Learner, self).__init__(config, policy)
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

    @tf.function
    def policy_forward_fn(self, obs_batch, act_batch, adv_batch, old_dist):
        with tf.GradientTape() as tape:
            old_logp_batch = tf.stop_gradient(old_dist.log_prob(act_batch))
            self.policy(obs_batch)
            a_dist = self.policy.actor.dist
            log_prob = a_dist.log_prob(act_batch)
            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob - old_logp_batch)
            surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            surrogate2 = adv_batch * ratio

            a_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            e_loss = tf.reduce_mean(a_dist.entropy())
            loss = a_loss - self.ent_coef * e_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                                for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                if grad is not None])
            else:
                if self.use_grad_clip:
                    self.optimizer.apply_gradients([(grad, var)
                                                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                    if grad is not None])
        return a_loss, e_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, ret_batch):
        with tf.GradientTape() as tape:
            _, _, v_pred, _ = self.policy(obs_batch)
            loss = tk.losses.mean_squared_error(ret_batch, v_pred)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                                for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                if grad is not None])
            else:
                if self.use_grad_clip:
                    self.optimizer.apply_gradients([(grad, var)
                                                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                    if grad is not None])
        return loss

    @tf.function
    def auxiliary_forward_fn(self, obs_batch, ret_batch, old_dist):
        with tf.GradientTape() as tape:
            _, _, v, aux_v = self.policy(obs_batch)
            a_dist = self.policy.actor.dist
            aux_loss = tk.losses.mean_squared_error(tf.stop_gradient(v), aux_v)
            kl_loss = tf.reduce_mean(a_dist.kl_divergence(old_dist))
            value_loss = tk.losses.mean_squared_error(ret_batch, v)
            loss = aux_loss + self.kl_beta * kl_loss + value_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                                for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                if grad is not None])
            else:
                if self.use_grad_clip:
                    self.optimizer.apply_gradients([(grad, var)
                                                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                                                    if grad is not None])
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
        obs_batch = samples['obs']
        act_batch = samples['actions']
        adv_batch = samples['advantages']
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        a_loss, e_loss = self.learn_policy(obs_batch, act_batch, adv_batch, old_dist)
        info = {"actor-loss": a_loss.numpy(), "entropy": e_loss.numpy()}
        self.policy_iterations += 1
        return info

    def update_critic(self, **samples):
        obs_batch = samples['obs']
        ret_batch = samples['returns']
        loss = self.learn_critic(obs_batch, ret_batch)
        info = {"critic-loss": loss.numpy()}
        self.value_iterations += 1
        return info

    def update_auxiliary(self, **samples):
        obs_batch = samples['obs']
        ret_batch = samples['returns']
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        loss = self.learn_auxiliary(obs_batch, ret_batch, old_dists)
        info = {"kl-loss": loss.numpy()}
        return info

    def update(self):
        pass
