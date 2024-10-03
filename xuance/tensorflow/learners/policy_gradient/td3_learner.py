"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(TD3_Learner, self).__init__(config, policy)
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
        self.actor_update_delay = config.actor_update_delay

    @tf.function
    def actor_forward_fn(self, obs_batch):
        with tf.GradientTape() as tape:
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -tf.reduce_mean(policy_q)
            gradients = tape.gradient(p_loss, self.policy.actor_trainable_variables)
            if self.use_grad_clip:
                self.optimizer['actor'].apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.actor_trainable_variables)
                    if grad is not None])
            else:
                self.optimizer['actor'].apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.actor_trainable_variables)
                    if grad is not None])
        return p_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, act_batch, rew_batch, next_batch, ter_batch):
        with tf.GradientTape() as tape:
            action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
            action_q_A = tf.reshape(action_q_A, [-1])
            action_q_B = tf.reshape(action_q_B, [-1])
            next_q = tf.reshape(self.policy.Qtarget(next_batch), [-1])
            target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
            target_q = tf.stop_gradient(tf.reshape(target_q, [-1]))
            q_loss = tk.losses.mean_squared_error(target_q, action_q_A) + tk.losses.mean_squared_error(target_q, action_q_B)
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
        return q_loss, action_q_A, action_q_B

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
            q_loss, action_q_A, action_q_B = self.policy.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, q_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q_A, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, action_q_B, axis=None))
        else:
            return self.critic_forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        info = {}
        obs_batch = samples['obs']
        act_batch = samples['actions']
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']

        q_loss, action_q_A, action_q_B = self.learn_critic(obs_batch, act_batch, rew_batch, next_batch, ter_batch)
        if self.iterations % self.actor_update_delay == 0:
            p_loss = self.learn_actor(obs_batch)
            self.policy.soft_update(self.tau)
            info["Ploss"] = p_loss.numpy()

        info.update({
            "Qloss": q_loss.numpy(),
            "QvalueA": tf.math.reduce_mean(action_q_A).numpy(),
            "QvalueB": tf.math.reduce_mean(action_q_B).numpy(),
        })

        return info
