from argparse import Action

from xuanpolicy.xuanpolicy_tf.learners import *
from xuanpolicy.xuanpolicy_tf.utils.operations import merge_distributions


class PPG_Learner(Learner):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25,
                 kl_beta: float = 1.0):
        super(PPG_Learner, self).__init__(policy, optimizer, summary_writer, device, modeldir)
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
            # Logger
            self.writer.add_scalar("actor-loss", a_loss.numpy(), self.policy_iterations)
            self.writer.add_scalar("entropy", e_loss.numpy(), self.policy_iterations)
            self.writer.add_scalar("lr_policy", lr_policy.numpy(), self.iterations)
            self.policy_iterations += 1

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
            self.writer.add_scalar("critic-loss", loss.numpy(), self.value_iterations)
            self.writer.add_scalar("lr_critic", lr_critic.numpy(), self.iterations)
            self.value_iterations += 1

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
            self.writer.add_scalar("kl-loss", loss.numpy(), self.value_iterations)
            self.writer.add_scalar("lr_aux", lr_aux.numpy(), self.iterations)

    def update(self):
        pass

    # def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
    #    #self.iterations += 1
    #     act_batch = torch.as_tensor(act_batch, device=self.device)
    #     ret_batch = torch.as_tensor(ret_batch, device=self.device)
    #     adv_batch = torch.as_tensor(adv_batch, device=self.device)
    #     old_logp_batch = torch.as_tensor(old_logp, device=self.device)
    #     outputs, a_dist, v_pred = self.policy(obs_batch)
    #     log_prob = a_dist.log_prob(act_batch)

    #     # ppo-clip core implementations 
    #     ratio = (log_prob - old_logp_batch).exp().float()
    #     surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
    #     surrogate2 = adv_batch * ratio
    #     a_loss = -torch.minimum(surrogate1, surrogate2).mean()
    #     c_loss = F.mse_loss(v_pred, ret_batch)
    #     e_loss = a_dist.entropy().mean()
    #     loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     if self.scheduler is not None:
    #         self.scheduler.step()
    #     # Logger
    #     lr = self.optimizer.state_dict()['param_groups'][0]['lr']
    #     cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
    #     self.writer.add_scalar("actor-loss", a_loss.item(), self.iterations)
    #     self.writer.add_scalar("critic-loss", c_loss.item(), self.iterations)
    #     self.writer.add_scalar("entropy", e_loss.item(), self.iterations)
    #     self.writer.add_scalar("learning_rate", lr, self.iterations)
    #     self.writer.add_scalar("predict_value", v_pred.mean().item(), self.iterations)
    #     self.writer.add_scalar("clip_ratio", cr, self.iterations)
