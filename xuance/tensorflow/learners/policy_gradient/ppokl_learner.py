from torch import kl_div
from xuance.tensorflow.learners import *
from xuance.tensorflow.utils.operations import merge_distributions


class PPOKL_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 target_kl: float = 0.25):
        super(PPOKL_Learner, self).__init__(policy, optimizer, device, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.kl_coef = 1.0

    def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_dists):
        self.iterations += 1
        with tf.device(self.device):
            act_batch = tf.convert_to_tensor(act_batch)
            ret_batch = tf.convert_to_tensor(ret_batch)
            adv_batch = tf.convert_to_tensor(adv_batch)

            with tf.GradientTape() as tape:
                outputs, _, v_pred = self.policy(obs_batch)
                a_dist = self.policy.actor.dist
                log_prob = a_dist.log_prob(act_batch)
                old_dist = merge_distributions(old_dists)
                kl = tf.reduce_mean(a_dist.kl_divergence(old_dist))
                old_logp_batch = old_dist.log_prob(act_batch)

                # ppo-clip core implementations
                ratio = tf.math.exp(log_prob - old_logp_batch)
                a_loss = -tf.reduce_mean(ratio * adv_batch) + self.kl_coef * kl
                c_loss = tk.losses.mean_squared_error(ret_batch, v_pred)
                e_loss = tf.reduce_mean(a_dist.entropy())
                loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
                if kl > self.target_kl * 1.5:
                    self.kl_coef = self.kl_coef * 2.
                elif kl < self.target_kl * 0.5:
                    self.kl_coef = self.kl_coef / 2.
                self.kl_coef = np.clip(self.kl_coef, 0.1, 20)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "actor-loss": a_loss.numpy(),
                "critic-loss": c_loss.numpy(),
                "entropy": e_loss.numpy(),
                "learning_rate": lr.numpy(),
                "kl": kl.numpy(),
                "predict_value": tf.math.reduce_mean(v_pred).numpy()
            }

            return info
