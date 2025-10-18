"""
Deep Deterministic Policy Gradient (DDPG)
Paper link: https://arxiv.org/pdf/1509.02971.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class DDPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(DDPG_Learner, self).__init__(config, policy, callback)
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
        self.mse_loss = tk.losses.MeanSquaredError()

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -tf.reduce_mean(policy_q)
            gradients = tape.gradient(p_loss, self.policy.actor_trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['actor'].apply_gradients(zip(gradients, self.policy.actor_trainable_variables))
            else:
                self.optimizer['actor'].apply_gradients(zip(gradients, self.policy.actor_trainable_variables))
        return p_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q = self.policy.Qaction(obs_batch, act_batch)
            next_q = self.policy.Qtarget(next_batch)
            backup = rew_batch + (1 - ter_batch) * self.gamma * next_q
            y_true = tf.reshape(tf.stop_gradient(backup), [-1])
            y_pred = tf.reshape(action_q, [-1])
            q_loss = self.mse_loss(y_true, y_pred)
            gradients = tape.gradient(q_loss, self.policy.critic_trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer['critic'].apply_gradients(zip(gradients, self.policy.critic_trainable_variables))
            else:
                self.optimizer['critic'].apply_gradients(zip(gradients, self.policy.critic_trainable_variables))
        return q_loss, action_q

    @tf.function
    def learn_actor(self, *inputs):
        if self.distributed_training:
            p_loss = self.policy.mirrored_strategy.run(self.actor_forward_fn, args=inputs)
            return self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, p_loss, axis=None)
        else:
            return self.actor_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            q_loss, action_q = self.policy.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q, axis=None))
        else:
            return self.critic_forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions']
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']
        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        # critic update
        q_loss, action_q = self.learn_critic(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        # actor update
        p_loss = self.learn_actor(obs_batch)

        self.policy.soft_update(self.tau)

        info.update({
            "Qloss": q_loss.numpy(),
            "Ploss": p_loss.numpy(),
            "Qvalue": tf.reduce_mean(action_q).numpy(),
        })

        info.update(self.callback.on_update_end(self.iterations, policy=self.policy, info=info,
                                                action_q=action_q, q_loss=q_loss, p_loss=p_loss))

        return info
