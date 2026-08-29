"""
Soft Actor-Critic with discrete action spaces (SAC-Discrete)
Paper link: https://arxiv.org/pdf/1910.07207.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class AlphaLayer(Module):
    def __init__(self, action_dim):
        super(AlphaLayer, self).__init__()
        self.log_alpha = self.add_weight(name="log_of_alpha", shape=(action_dim,), initializer=tf.zeros, trainable=True)


class SACDIS_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(SACDIS_Learner, self).__init__(config, model, callback)
        if self.distributed_training:
            with self.model.mirrored_strategy.scope():
                self.optimizer = {'actor': keras.optimizers.Adam(config.learning_rate_actor),
                                  'critic': keras.optimizers.Adam(config.learning_rate_critic)}
        else:
            self.optimizer = {'actor': keras.optimizers.Adam(config.learning_rate_actor),
                              'critic': keras.optimizers.Adam(config.learning_rate_critic)}
        self.tau = config.tau
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        self.mse_loss = keras.losses.MeanSquaredError()
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -float(model.actor.action_space.n)
            if self.distributed_training:
                with self.model.mirrored_strategy.scope():
                    self.alpha_layer = AlphaLayer(1)
                    self.alpha = tf.exp(self.alpha_layer.log_alpha)
                    self.alpha_optimizer = keras.optimizers.Adam(config.learning_rate_actor)
            else:
                self.alpha_layer = AlphaLayer(1)
                self.alpha = tf.exp(self.alpha_layer.log_alpha)
                self.alpha_optimizer = keras.optimizers.Adam(config.learning_rate_actor)

    def current_alpha(self, dtype):
        if self.use_automatic_entropy_tuning:
            alpha = tf.exp(self.alpha_layer.log_alpha)
        else:
            alpha = tf.convert_to_tensor(self.alpha)

        return tf.stop_gradient(tf.cast(alpha, dtype))

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            action_prob, log_pi, policy_q_1, policy_q_2 = self.model.Qpolicy(obs_batch)
            policy_q = tf.math.minimum(policy_q_1, policy_q_2)
            alpha = self.current_alpha(log_pi.dtype)
            p_loss = tf.reduce_mean(tf.reduce_sum(action_prob * (alpha * log_pi - policy_q), axis=-1))
            gradients = tape.gradient(p_loss, self.model.actor.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['actor'].apply_gradients(zip(gradients, self.model.actor.trainable_variables))
            else:
                self.optimizer['actor'].apply_gradients(zip(gradients, self.model.actor.trainable_variables))
        return p_loss, log_pi, policy_q

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q_1, action_q_2 = self.model.Qaction(obs_batch)
            action_q_1 = tf.gather(params=action_q_1, indices=act_batch, axis=-1, batch_dims=-1)
            action_q_2 = tf.gather(params=action_q_2, indices=act_batch, axis=-1, batch_dims=-1)
            action_prob_next, log_pi_next, target_q = self.model.Qtarget(next_batch)
            alpha = self.current_alpha(log_pi_next.dtype)
            target_q = action_prob_next * (target_q - alpha * log_pi_next)
            target_q = tf.reduce_sum(target_q, axis=1)
            backup = rew_batch + (1 - ter_batch) * self.gamma * target_q
            backup = tf.stop_gradient(backup)
            q_loss = self.mse_loss(backup, action_q_1) + self.mse_loss(backup, action_q_2)
            gradients = tape.gradient(q_loss, self.model.critic.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['critic'].apply_gradients(zip(gradients, self.model.critic.trainable_variables))
            else:
                self.optimizer['critic'].apply_gradients(zip(gradients, self.model.critic.trainable_variables))
        return q_loss

    @tf.function
    def alpha_forward_fn(self, log_pi):
        with tf.GradientTape() as tape:
            alpha_loss = -tf.math.reduce_mean(self.alpha_layer.log_alpha.value() * (log_pi + self.target_entropy))
            gradients = tape.gradient(alpha_loss, self.alpha_layer.trainable_variables)
            self.alpha_optimizer.apply_gradients(zip(gradients, self.alpha_layer.trainable_variables))
        return alpha_loss

    @tf.function
    def learn_actor(self, *inputs):
        if self.distributed_training:
            p_loss, log_pi, policy_q = self.model.mirrored_strategy.run(self.actor_forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, p_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, log_pi, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, policy_q, axis=None))
        else:
            return self.actor_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            q_loss = self.model.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None)
        else:
            return self.critic_forward_fn(*inputs)

    @tf.function
    def learn_alpha(self, *inputs):
        if self.distributed_training:
            alpha_loss = self.model.mirrored_strategy.run(self.alpha_forward_fn, args=inputs)
            return self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, alpha_loss, axis=None)
        else:
            return self.alpha_forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], tf.float32)
        act_batch = tf.expand_dims(tf.convert_to_tensor(samples['actions'], tf.int32), axis=-1)
        next_batch = tf.convert_to_tensor(samples['obs_next'], tf.float32)
        rew_batch = tf.convert_to_tensor(samples['rewards'], tf.float32)
        ter_batch = tf.convert_to_tensor(samples['terminals'], tf.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        q_loss = self.learn_critic(obs_batch, act_batch, rew_batch, next_batch, ter_batch)
        p_loss, log_pi, policy_q = self.learn_actor(obs_batch)
        if self.use_automatic_entropy_tuning:
            alpha_loss = self.learn_alpha(log_pi)
            alpha_loss = alpha_loss
            self.alpha = tf.math.exp(self.alpha_layer.log_alpha)
        else:
            alpha_loss = 0

        self.model.soft_update(self.tau)

        info.update({
            "Qloss": q_loss,
            "Ploss": p_loss,
            "Qvalue": tf.reduce_mean(policy_q),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                log_pi=log_pi, policy_q=policy_q, p_loss=p_loss, q_loss=q_loss,
                                                alpha_loss=alpha_loss, alpha=self.alpha))

        return info
