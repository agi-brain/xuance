"""
Deep Deterministic Policy Gradient (DDPG)
Paper link: https://arxiv.org/pdf/1509.02971.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class DDPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(DDPG_Learner, self).__init__(config, model, callback)
        self.optimizer = {'actor': keras.optimizers.Adam(config.learning_rate_actor),
                          'critic': keras.optimizers.Adam(config.learning_rate_critic)}
        self.tau = config.tau
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            model_q = self.model.Qpolicy(obs_batch)
            p_loss = -tf.reduce_mean(model_q)
            gradients = tape.gradient(p_loss, self.model.actor.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['actor'].apply_gradients(zip(gradients, self.model.actor.trainable_variables))
            else:
                self.optimizer['actor'].apply_gradients(zip(gradients, self.model.actor.trainable_variables))
        return p_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q = tf.reshape(self.model.Qaction(obs_batch, act_batch), [-1])
            next_q = tf.reshape(self.model.Qtarget(next_batch), [-1])
            target_q = rew_batch + (1 - ter_batch) * self.gamma * next_q
            q_loss = self.mse_loss(tf.stop_gradient(target_q), action_q)
            gradients = tape.gradient(q_loss, self.model.critic.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['critic'].apply_gradients(zip(gradients, self.model.critic.trainable_variables))
            else:
                self.optimizer['critic'].apply_gradients(zip(gradients, self.model.critic.trainable_variables))
        return q_loss, action_q

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
            q_loss, action_q = self.model.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q, axis=None))
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

        # critic update
        q_loss, action_q = self.learn_critic(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        # actor update
        p_loss = self.learn_actor(obs_batch)

        self.model.soft_update(self.tau)

        info.update({
            "Qloss": q_loss,
            "Ploss": p_loss,
            "Qvalue": tf.reduce_mean(action_q),
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                action_q=action_q, q_loss=q_loss, p_loss=p_loss))

        return info
