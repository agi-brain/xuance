"""
Soft Actor-Critic with discrete action spaces (SAC-Discrete)
Paper link: https://arxiv.org/pdf/1910.07207.pdf
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class AlphaLayer(Module):
    def __init__(self, action_dim):
        super(AlphaLayer, self).__init__()
        self.log_alpha = self.add_weight(name="log_of_alpha", shape=(action_dim,), initializer=tf.zeros, trainable=True)


class SACDIS_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 target_entropy: Optional[float] = None):
        super(SACDIS_Learner, self).__init__(config, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = {'actor': tk.optimizers.legacy.Adam(config.learning_rate_actor),
                                      'critic': tk.optimizers.legacy.Adam(config.learning_rate_critic)}
            else:
                self.optimizer = {'actor': tk.optimizers.legacy.Adam(config.learning_rate_actor),
                                  'critic': tk.optimizers.legacy.Adam(config.learning_rate_critic)}
        else:
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = {'actor': tk.optimizers.Adam(config.learning_rate_actor),
                                      'critic': tk.optimizers.Adam(config.learning_rate_critic)}
            else:
                self.optimizer = {'actor': tk.optimizers.Adam(config.learning_rate_actor),
                                  'critic': tk.optimizers.Adam(config.learning_rate_critic)}
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = target_entropy
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.alpha_layer = AlphaLayer(1)
                    self.alpha = tf.exp(self.alpha_layer.log_alpha)
                    if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
                        self.alpha_optimizer = tk.optimizers.legacy.Adam(config.learning_rate_actor)
                    else:
                        self.alpha_optimizer = tk.optimizers.Adam(config.learning_rate_actor)
            else:
                self.alpha_layer = AlphaLayer(1)
                self.alpha = tf.exp(self.alpha_layer.log_alpha)
                if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
                    self.alpha_optimizer = tk.optimizers.legacy.Adam(config.learning_rate_actor)
                else:
                    self.alpha_optimizer = tk.optimizers.Adam(config.learning_rate_actor)

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            action_prob, log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
            policy_q = tf.math.minimum(policy_q_1, policy_q_2)
            p_loss = tf.reduce_mean(tf.reduce_sum(action_prob * (self.alpha * log_pi - policy_q), axis=-1))
            gradients = tape.gradient(p_loss, self.policy.actor_trainable_variables)
            if self.use_grad_clip:
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(
                        gradients, self.policy.actor_trainable_variables)
                    if grad is not None])
            else:
                self.optimizer['actor'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(
                        gradients, self.policy.actor_trainable_variables)
                    if grad is not None])
        return p_loss, log_pi, policy_q

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q_1, action_q_2 = self.policy.Qaction(obs_batch)
            action_q_1 = tf.gather(params=action_q_1, indices=act_batch, axis=-1, batch_dims=-1)
            action_q_2 = tf.gather(params=action_q_2, indices=act_batch, axis=-1, batch_dims=-1)
            action_prob_next, log_pi_next, target_q = self.policy.Qtarget(next_batch)
            target_q = action_prob_next * (target_q - self.alpha * log_pi_next)
            target_q = tf.expand_dims(tf.reduce_sum(target_q, axis=1), axis=-1)
            rew = tf.expand_dims(rew_batch, axis=-1)
            backup = rew + (1 - ter_batch) * self.gamma * target_q
            y_true = tf.stop_gradient(tf.reshape(backup, [-1]))
            y_pred_1 = tf.reshape(action_q_1, [-1])
            y_pred_2 = tf.reshape(action_q_2, [-1])
            q_loss = tk.losses.mean_squared_error(y_true, y_pred_1) + tk.losses.mean_squared_error(y_true, y_pred_2)
            gradients = tape.gradient(q_loss, self.policy.critic_trainable_variables)
            if self.use_grad_clip:
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.critic_trainable_variables)
                    if grad is not None])
            else:
                self.optimizer['critic'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.critic_trainable_variables)
                    if grad is not None])
        return q_loss

    @tf.function
    def alpha_forward_fn(self, log_pi):
        with tf.GradientTape() as tape:
            alpha_loss = -tf.math.reduce_mean(self.alpha_layer.log_alpha.value() * (log_pi + self.target_entropy))
            gradients = tape.gradient(alpha_loss, self.alpha_layer.trainable_variables)
            self.alpha_optimizer.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.alpha_layer.trainable_variables)
                if grad is not None
            ])
        return alpha_loss

    @tf.function
    def learn_actor(self, *inputs):
        if self.distributed_training:
            p_loss, log_pi, policy_q = self.policy.mirrored_strategy.run(self.actor_forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, p_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, log_pi, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, policy_q, axis=None))
        else:
            return self.actor_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            q_loss = self.policy.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None)
        else:
            return self.critic_forward_fn(*inputs)

    @tf.function
    def learn_alpha(self, *inputs):
        if self.distributed_training:
            alpha_loss = self.policy.mirrored_strategy.run(self.alpha_forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, alpha_loss, axis=None)
        else:
            return self.alpha_forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions'].astype(np.int32)
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']

        q_loss = self.learn_critic(obs_batch, act_batch, rew_batch, next_batch, ter_batch)
        p_loss, log_pi, policy_q = self.learn_actor(obs_batch)
        if self.use_automatic_entropy_tuning:
            alpha_loss = self.learn_alpha(log_pi)
            alpha_loss = alpha_loss.numpy()
            self.alpha = tf.math.exp(self.alpha_layer.log_alpha).numpy()
        else:
            alpha_loss = 0

        self.policy.soft_update(self.tau)

        info = {
            "Qloss": q_loss.numpy(),
            "Ploss": p_loss.numpy(),
            "Qvalue": tf.reduce_mean(policy_q).numpy(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }

        return info
