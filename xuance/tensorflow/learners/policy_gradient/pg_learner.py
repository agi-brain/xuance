"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(PG_Learner, self).__init__(config, model, callback)
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.ent_coef = config.ent_coef

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch):
        with tf.GradientTape() as tape:
            a_dist = self.model(obs_batch).distributions
            log_prob = a_dist.log_prob(act_batch)

            a_loss = -tf.reduce_mean(ret_batch * log_prob)
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
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, e_loss = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        ret_batch = tf.convert_to_tensor(samples['returns'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples["actions"], dtype=tf.float32)

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch, returns=ret_batch)

        a_loss, e_loss = self.learn(obs_batch, act_batch, ret_batch)

        info.update({
            "actor-loss": a_loss,
            "entropy": e_loss
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                a_loss=a_loss, e_loss=e_loss))

        return info
