"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(TD3_Learner, self).__init__(config, model, callback)
        self.optimizer = {'actor': keras.optimizers.Adam(config.learning_rate_actor),
                          'critic': keras.optimizers.Adam(config.learning_rate_critic)}
        self.tau = config.tau
        self.actor_update_delay = config.actor_update_delay
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            policy_q = self.model.Qpolicy(obs_batch)
            p_loss = -tf.reduce_mean(policy_q)
            gradients = tape.gradient(p_loss, self.model.actor.trainable_variables)
            if self.use_grad_clip:
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.model.actor.trainable_variables)
                    if grad is not None])
            else:
                self.optimizer['actor'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.model.actor.trainable_variables)
                    if grad is not None])
        return p_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q_A, action_q_B = self.model.Qaction(obs_batch, act_batch)
            action_q_A = tf.reshape(action_q_A, [-1])
            action_q_B = tf.reshape(action_q_B, [-1])
            next_q = tf.reshape(self.model.Qtarget(next_batch), [-1])
            target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
            target_q = tf.stop_gradient(target_q)
            q_loss = self.mse_loss(target_q, action_q_A) + self.mse_loss(target_q, action_q_B)
            gradients = tape.gradient(q_loss, self.model.critic.trainable_variables)
            if self.use_grad_clip:
                self.optimizer['critic'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.model.critic.trainable_variables)
                    if grad is not None])
            else:
                self.optimizer['critic'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.model.critic.trainable_variables)
                    if grad is not None])
        return q_loss, action_q_A, action_q_B

    @tf.function
    def learn_actor(self, *inputs):
        if self.distributed_training:
            p_loss = self.model.mirrored_strategy.run(self.actor_forward_fn, args=inputs)
            return self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, p_loss, axis=None)
        else:
            return self.actor_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            q_loss, action_q_A, action_q_B = self.model.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q_A, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q_B, axis=None))
        else:
            return self.critic_forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.float32)
        next_batch = tf.convert_to_tensor(samples['obs_next'], dtype=tf.float32)
        rew_batch = tf.convert_to_tensor(samples['rewards'], dtype=tf.float32)
        ter_batch = tf.convert_to_tensor(samples['terminals'], dtype=tf.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        q_loss, action_q_A, action_q_B = self.learn_critic(obs_batch, act_batch, rew_batch, next_batch, ter_batch)

        if self.iterations % self.actor_update_delay == 0:
            p_loss = self.learn_actor(obs_batch)
            self.model.soft_update(self.tau)
            info["Ploss"] = p_loss

        info.update({
            "Qloss": q_loss,
            "QvalueA": tf.math.reduce_mean(action_q_A),
            "QvalueB": tf.math.reduce_mean(action_q_B),
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                action_q_A=action_q_A, action_q_B=action_q_B, q_loss=q_loss))

        return info
