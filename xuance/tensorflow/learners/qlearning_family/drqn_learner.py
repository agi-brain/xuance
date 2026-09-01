"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(DRQN_Learner, self).__init__(config, model, callback)
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.sync_frequency = config.sync_frequency
        self.n_actions = self.model.n_actions
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def forward_fn(self, batch_size, obs_batch, act_batch, rew_batch, ter_batch):
        with tf.GradientTape() as tape:
            rnn_states = self.model.init_rnn_states(batch_size)
            _, model_output = self.model(obs_batch[:, 0:-1], rnn_states=rnn_states)
            evalQ = model_output.values
            _, target_model_output = self.model.target(obs_batch, rnn_states=rnn_states)
            targetA, targetQ = target_model_output.actions, target_model_output.values
            targetA = targetA[:, 1:]
            targetQ = targetQ[:, 1:]
            # targetQ = targetQ.max(dim=-1).values

            predictQ = tf.gather(evalQ, act_batch, axis=-1, batch_dims=2)
            targetQ = tf.gather(targetQ, targetA, axis=-1, batch_dims=2)
            targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
            targetQ = tf.stop_gradient(targetQ)

            loss = self.mse_loss(targetQ, predictQ)
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

        return predictQ, loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            predictQ, loss = self.model.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, predictQ, axis=None),
                    self.model.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
        act_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.int32)
        rew_batch = tf.convert_to_tensor(samples['rewards'], dtype=tf.float32)
        ter_batch = tf.convert_to_tensor(samples['terminals'], dtype=tf.float32)
        batch_size = samples['batch_size']
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             rew=rew_batch, termination=ter_batch, batch_size=batch_size)

        predictQ, loss = self.learn(batch_size, obs_batch, act_batch, rew_batch, ter_batch)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        info.update({
            "Qloss": loss,
            "predictQ": tf.math.reduce_mean(predictQ)
        })

        info.update(self.callback.on_update_end(self.iterations, model=self.model, info=info,
                                                predictQ=predictQ, loss=loss))

        return info
