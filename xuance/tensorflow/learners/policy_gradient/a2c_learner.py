"""
Advantage Actor-Critic (A2C)
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(A2C_Learner, self).__init__(config, model, callback)
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch, adv_batch):
        with tf.GradientTape() as tape:
            model_output = self.model(obs_batch)
            a_dist = model_output.distributions
            v_pred = model_output.values
            log_prob = a_dist.log_prob(act_batch)

            a_loss = -tf.reduce_mean(adv_batch * log_prob)
            c_loss = self.mse_loss(ret_batch, v_pred)
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
        obs_batch = tf.convert_to_tensor(samples["obs"], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples["returns"], dtype=tf.float32)
        adv_batch = tf.convert_to_tensor(samples['advantages'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch)

        a_loss, c_loss, e_loss, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch)

        info.update({
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy()
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                v_pred=v_pred, a_loss=a_loss, c_loss=c_loss, e_loss=e_loss))

        return info
