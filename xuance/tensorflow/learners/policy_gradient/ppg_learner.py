from xuance.tensorflow.learners import *
from xuance.tensorflow.utils.operations import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25,
                 kl_beta: float = 1.0):
        super(PPG_Learner, self).__init__(policy, optimizer, device, model_dir)
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.kl_beta = kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0

    def update_policy(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch)
            ret_batch = tf.convert_to_tensor(ret_batch)
            adv_batch = tf.convert_to_tensor(adv_batch)

            with tf.GradientTape() as tape:
                old_dist = merge_distributions(old_dists)
                old_logp_batch = tf.stop_gradient(old_dist.log_prob(act_batch))

                outputs, _, _, _ = self.policy(obs_batch)
                a_dist = self.policy.actor.dist
                log_prob = a_dist.log_prob(act_batch)
                # ppo-clip core implementations
                ratio = tf.math.exp(log_prob - old_logp_batch)
                surrogate1 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
                surrogate2 = adv_batch * ratio

                a_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                e_loss = tf.reduce_mean(a_dist.entropy())
                loss = a_loss - self.ent_coef * e_loss
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            lr_policy = self.optimizer._decayed_lr(tf.float32)

            info = {
                "actor-loss": a_loss.numpy(),
                "entropy": e_loss.numpy(),
                "learning_rate": lr_policy.numpy(),
            }
            self.policy_iterations += 1

            return info

    def update_critic(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        with tf.device(self.device):
            ret_batch = tf.convert_to_tensor(ret_batch)
            with tf.GradientTape() as tape:
                _, _, v_pred, _ = self.policy(obs_batch)
                loss = tk.losses.mean_squared_error(ret_batch, v_pred)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            lr_critic = self.optimizer._decayed_lr(tf.float32)
            info = {
                "critic-loss": loss.numpy(),
                "lr_critic": lr_critic.numpy()
            }
            self.value_iterations += 1
            return info

    def update_auxiliary(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch)
            ret_batch = tf.convert_to_tensor(ret_batch)
            adv_batch = tf.convert_to_tensor(adv_batch)

            with tf.GradientTape() as tape:
                old_dist = merge_distributions(old_dists)
                outputs, _, v, aux_v = self.policy(obs_batch)
                a_dist = self.policy.actor.dist
                aux_loss = tk.losses.mean_squared_error(tf.stop_gradient(v), aux_v)
                kl_loss = tf.reduce_mean(a_dist.kl_divergence(old_dist))
                value_loss = tk.losses.mean_squared_error(ret_batch, v)
                loss = aux_loss + self.kl_beta * kl_loss + value_loss
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            lr_aux = self.optimizer._decayed_lr(tf.float32)

            info = {
                "kl-loss": loss.numpy(),
                "lr_aux": lr_aux.numpy()
            }
            return info

    def update(self):
        pass
