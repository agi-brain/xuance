"""
Proximal Policy Optimization (PPO) with clip trick
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class PPO_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(PPO_Learner, self).__init__(config, model, callback)
        self.scheduler = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=self.total_iters,
            end_learning_rate=config.learning_rate * self.end_factor_lr_decay,
            power=1.0  # 1.0 indicates linear decay.
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=self.scheduler, epsilon=1e-5)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        buffer_size = self.config.horizon_size * self.config.parallels
        update_times = self.config.running_steps // buffer_size
        total_iters = update_times * self.config.n_epochs * self.config.n_minibatch
        return total_iters

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
        with tf.GradientTape() as tape:
            model_output = self.model(obs_batch)
            a_dist = model_output.distributions
            v_pred = model_output.values
            log_prob = a_dist.log_prob(act_batch)

            # ppo-clip core implementations
            ratio = tf.math.exp(log_prob - old_logp)
            surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            surrogate2 = adv_batch * ratio

            a_loss = -tf.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
            c_loss = tf.reduce_mean(tf.square(v_pred - ret_batch))
            e_loss = tf.reduce_mean(a_dist.entropy())

            loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.use_grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip_norm)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return a_loss, c_loss, e_loss, v_pred

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, c_loss, e_loss, v_pred = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, c_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, v_pred, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples['returns'], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'], dtype=tf.float32)
        old_logp_batch = tf.convert_to_tensor(samples['aux_batch']['old_logp'], dtype=tf.float32)
        
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch, old_logp=old_logp_batch)

        a_loss, c_loss, e_loss, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch, old_logp_batch)

        info.update({
            "actor-loss": a_loss,
            "critic-loss": c_loss,
            "entropy": e_loss,
            "predict_value": tf.math.reduce_mean(v_pred),
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                v_pred=v_pred, a_loss=a_loss, c_loss=c_loss, e_loss=e_loss))

        return info
