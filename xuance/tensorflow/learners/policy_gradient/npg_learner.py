from argparse import Namespace
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class NPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(NPG_Learner, self).__init__(config, model, callback)
        self.actor_optimizer = keras.optimizers.Adam(config.learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(config.learning_rate)
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch):
        with tf.GradientTape() as critic_tape:
            model_output = self.model.critic(obs_batch)
            v_pred = model_output.values
            c_loss = self.mse_loss(ret_batch, v_pred)

        critic_variables = self.model.critic.trainable_variables
        critic_grads = critic_tape.gradient(c_loss, critic_variables)

        if self.use_grad_clip:
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.grad_clip_norm)

        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

        with tf.GradientTape() as actor_tape:
            model_output = self.model.actor(obs_batch)
            a_dist = model_output.distributions
            log_prob = a_dist.log_prob(act_batch)
            a_loss = -tf.reduce_mean(adv_batch * log_prob)

        actor_variables = self.model.actor.trainable_variables
        actor_grads = actor_tape.gradient(a_loss, actor_variables)

        # Natural policy gradient.
        natural_grads = []
        for param, grad in zip(actor_variables, actor_grads):
            if grad is None:
                natural_grads.append(None)
                continue

            fisher_inv = self.compute_fisher_information([param], obs_batch, act_batch)
            grad_flat = tf.reshape(grad, [-1])
            natural_grad = tf.linalg.matvec(fisher_inv, grad_flat)
            natural_grad = tf.reshape(natural_grad, tf.shape(param))
            natural_grads.append(natural_grad)

        if self.use_grad_clip:
            # clip_by_global_norm cannot safely handle None directly.
            valid_indices = [i for i, grad in enumerate(natural_grads) if grad is not None]
            valid_grads = [natural_grads[i] for i in valid_indices]
            clipped_grads, _ = tf.clip_by_global_norm(valid_grads, self.grad_clip_norm)
            for i, grad in zip(valid_indices, clipped_grads):
                natural_grads[i] = grad

        self.actor_optimizer.apply_gradients(
            [(grad, var) for grad, var in zip(natural_grads, actor_variables) if grad is not None])

        return a_loss, c_loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, c_loss = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples['returns'], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'], dtype=tf.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch)

        a_loss, c_loss = self.learn(obs_batch, act_batch, ret_batch, adv_batch)

        info.update({
            "actor-loss": a_loss,
            "critic-loss": c_loss
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                a_loss=a_loss, c_loss=c_loss))

        return info

    def compute_fisher_information(self, params, obs, act):
        # Total number of parameters.
        param_num = sum(tf.size(param) for param in params)

        with tf.GradientTape() as tape:
            dist = self.model(obs).distributions
            log_probs = dist.log_prob(act)
            log_prob_sum = tf.reduce_sum(log_probs)

        # Gradients for all parameters.
        grads = tape.gradient(log_prob_sum, params)

        # Flatten and concatenate all gradients.
        score = tf.concat([tf.reshape(grad, [-1]) for grad in grads if grad is not None], axis=0)

        # Fisher information matrix.
        fisher_information = tf.tensordot(score, score, axes=0)

        fisher_information *= log_prob_sum

        fisher_information /= self.config.horizon_size

        # Damping term.
        fisher_information += 1e-3 * tf.eye(param_num, dtype=fisher_information.dtype)

        fisher_inv = tf.linalg.inv(fisher_information)

        return fisher_inv
