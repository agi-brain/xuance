"""
Parameterised deep Q network (P-DQN)
Paper link: https://arxiv.org/pdf/1810.06394.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import tf, keras, Module
from xuance.tensorflow.learners import Learner


class PDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(PDQN_Learner, self).__init__(config, model, callback)
        continuous_actor_optimizer = keras.optimizers.Adam(config.learning_rate)
        q_network_optimizer = keras.optimizers.Adam(config.learning_rate)
        self.optimizer = [continuous_actor_optimizer, q_network_optimizer]
        self.tau = config.tau
        self.mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def _forward_actor(self, obs_batch):
        # optimize actor network
        with tf.GradientTape() as tape:
            policy_q = self.model.Qpolicy(obs_batch)
            p_loss = -tf.reduce_mean(policy_q)
            gradients = tape.gradient(p_loss, self.model.continuous_actor.trainable_variables)
            self.optimizer[0].apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.model.continuous_actor.trainable_variables)
                if grad is not None
            ])
        return p_loss

    @tf.function
    def _forward_critic(self, obs_batch, disact_batch, conact_batch, next_batch, rew_batch, ter_batch):
        # optimize Q-network
        with tf.GradientTape() as tape:
            target_conact = self.model.Atarget(next_batch)
            target_q = self.model.Qtarget(next_batch, target_conact)
            target_q = tf.squeeze(tf.reduce_max(target_q, axis=1, keepdims=True)[0])

            target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

            eval_qs = self.model.Qeval(obs_batch, conact_batch)
            eval_q = tf.gather(eval_qs, tf.reshape(disact_batch, [-1, 1]), axis=-1, batch_dims=-1)
            y_true = tf.reshape(tf.stop_gradient(target_q), [-1])
            y_pred = tf.reshape(eval_q, [-1])
            q_loss = self.mse_loss(y_true, y_pred)

            gradients = tape.gradient(q_loss, self.model.q_network.trainable_variables)
            self.optimizer[1].apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.model.q_network.trainable_variables)
                if grad is not None
            ])
        return eval_q, q_loss

    @tf.function
    def _learn_critic(self, *inputs):
        if self.distributed_training:
            strategy = tf.distribute.get_strategy()
            eval_q, q_loss = strategy.run(self._forward_critic, args=inputs)
            return (strategy.reduce(tf.distribute.ReduceOp.MEAN, eval_q, axis=None),
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, q_loss, axis=None))
        else:
            return self._forward_critic(*inputs)

    @tf.function
    def _learn_actor(self, *inputs):
        if self.distributed_training:
            strategy = tf.distribute.get_strategy()
            p_loss = strategy.run(self._forward_actor, args=inputs)
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, p_loss, axis=None)
        else:
            return self._forward_actor(*inputs)

    def update(self, **samples):
        self.iterations += 1
        with tf.device(self.device):
            obs_batch = tf.convert_to_tensor(samples['obs'], dtype=tf.float32)
            hyact_batch = tf.convert_to_tensor(samples['actions'], dtype=tf.float32)
            next_batch = tf.convert_to_tensor(samples['obs_next'], dtype=tf.float32)
            rew_batch = tf.convert_to_tensor(samples['rewards'], dtype=tf.float32)
            ter_batch = tf.convert_to_tensor(samples['terminals'], dtype=tf.float32)
            disact_batch = tf.cast(hyact_batch[:, 0], dtype=tf.int32)
            conact_batch = hyact_batch[:, 1:]

            eval_q, q_loss = self._learn_critic(obs_batch, disact_batch, conact_batch, next_batch, rew_batch, ter_batch)

            p_loss = self._learn_actor(obs_batch)

            self.model.soft_update(self.tau)

            info = {
                "Q_loss": q_loss,
                "P_loss": p_loss,
                'Qvalue': tf.math.reduce_mean(eval_q)
            }

            return info
