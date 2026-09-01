"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner
from xuance.tensorflow.rl_models.modules import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(PPG_Learner, self).__init__(config, model, callback)
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def model_forward_fn(self, obs_batch, act_batch, adv_batch, old_log_prob_batch):
        with tf.GradientTape() as tape:
            a_dist = self.model.actor(obs_batch).distributions
            log_prob = a_dist.log_prob(act_batch)
            ratio = tf.math.exp(log_prob - old_log_prob_batch)
            surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            surrogate2 = adv_batch * ratio

            a_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            e_loss = tf.reduce_mean(a_dist.entropy())

            loss = a_loss - self.ent_coef * e_loss
            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return a_loss, e_loss

    @tf.function
    def critic_forward_fn(self, obs_batch, ret_batch):
        with tf.GradientTape() as tape:
            v_pred = self.model.critic(obs_batch).values

            loss = self.mse_loss(ret_batch, v_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def auxiliary_forward_fn(self, *args):
        with tf.GradientTape() as tape:
            obs_batch, ret_batch, old_dists = args

            model_output = self.model(obs_batch)
            a_dist, v = model_output.distributions, model_output.values
            aux_v = self.model.aux_critic(obs_batch).values

            aux_loss = self.mse_loss(tf.stop_gradient(v), aux_v)
            kl_loss = tf.reduce_mean(a_dist.kl_divergence(old_dists))
            value_loss = self.mse_loss(ret_batch, v)

            loss = aux_loss + self.kl_beta * kl_loss + value_loss
            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def learn_model(self, *inputs):
        if self.distributed_training:
            a_loss, e_loss = self.model.mirrored_strategy.run(self.model_forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None))
        else:
            return self.model_forward_fn(*inputs)

    @tf.function
    def learn_critic(self, *inputs):
        if self.distributed_training:
            loss = self.model.mirrored_strategy.run(self.critic_forward_fn, args=inputs)
            return self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        else:
            return self.critic_forward_fn(*inputs)

    @tf.function
    def learn_auxiliary(self, *inputs):
        if self.distributed_training:
            loss = self.model.mirrored_strategy.run(self.auxiliary_forward_fn, args=inputs)
            return self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        else:
            return self.auxiliary_forward_fn(*inputs)

    def update_actor(self, **samples):
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_log_prob_batch = tf.stop_gradient(old_dist.log_prob(act_batch))

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch, advantages=adv_batch,
                                             old_dist=old_dist, old_logp=old_log_prob_batch)

        a_loss, e_loss = self.learn_model(obs_batch, act_batch, adv_batch, old_log_prob_batch)
        info.update({"actor-loss": a_loss.numpy(), "entropy": e_loss.numpy()})
        self.policy_iterations += 1
        info.update(self.callback.on_update_end(self.iterations, method="update_model",
                                                model=self.model, info=info,
                                                a_loss=a_loss, e_loss=e_loss))
        return info

    def update_critic(self, **samples):
        self.value_iterations += 1
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        info = self.callback.on_update_start(self.iterations, model=self.model, obs=obs_batch, returns=ret_batch)

        loss = self.learn_critic(obs_batch, ret_batch)
        info.update({"critic-loss": loss.numpy()})
        info.update(self.callback.on_update_end(self.iterations, method="update_critic",
                                                model=self.model, info=info, loss=loss))
        return info

    def update_auxiliary_critic(self, **samples):
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        old_dists = merge_distributions(samples['aux_batch']['old_dist'])
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, returns=ret_batch, old_dist=old_dists)

        loss = self.learn_auxiliary(obs_batch, ret_batch, old_dists)
        info.update({"kl-loss": loss.numpy()})
        info.update(self.callback.on_update_end(self.iterations, method="update_auxiliary",
                                                model=self.model, info=info, loss=loss))
        return info

    def update(self, *args):
        pass
