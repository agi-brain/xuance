"""
DQN with Quantile Regression (QRDQN)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11791
Implementation: TensorFlow2
"""
import numpy as np
from argparse import Namespace
from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class QRDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(QRDQN_Learner, self).__init__(config, model, callback)
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            evalZ = self.model(obs_batch).values
            target_model_output = self.model.target(next_batch)
            targetA = target_model_output.actions
            targetZ = target_model_output.values
            current_quantile = tf.math.reduce_sum(
                evalZ * tf.expand_dims(tf.one_hot(act_batch, evalZ.shape[1]), axis=-1), axis=1)
            target_quantile = tf.math.reduce_sum(targetZ * tf.expand_dims(tf.one_hot(targetA, evalZ.shape[1]), axis=-1),
                                                 axis=1)
            target_quantile = tf.expand_dims(rew_batch, 1) + self.gamma * target_quantile * (
                        1 - tf.expand_dims(ter_batch, 1))
            target_quantile = tf.stop_gradient(target_quantile)
            loss = self.mse_loss(tf.reshape(target_quantile, [-1, ]),
                                                tf.reshape(current_quantile, [-1, ]))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if self.use_grad_clip:
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.model.trainable_variables)
                    if grad is not None
                ])
            else:
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.model.trainable_variables)
                    if grad is not None
                ])
        return current_quantile, loss

    def learn(self, *inputs):
        if self.distributed_training:
            predictQ, loss = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions'].astype(np.int32)
        next_batch = samples['obs_next']
        rew_batch = samples['rewards']
        ter_batch = samples['terminals']
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        current_quantile, loss = self.learn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update({
            "Qloss": loss.numpy(),
            "predictQ": tf.math.reduce_mean(current_quantile).numpy()
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                current_quantile=current_quantile, loss=loss))

        return info
