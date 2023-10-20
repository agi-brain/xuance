"""
Value Decomposition Actor-Critic (VDAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: Pytorch
"""
from xuance.tensorflow.learners import *


class VDAC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        super(VDAC_Learner, self).__init__(config, policy, optimizer, device, modeldir)

    def save_model(self):
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        self.policy.actor.save_weights(model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.sort()
            model_path = path + model_names[-1]
            self.policy.actor.load_weights(model_path)
        except:
            raise "Failed to load model! Please train and save the model first."

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'])
            returns = tf.reduce_mean(tf.convert_to_tensor(sample['values']), axis=1)
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
            advantages = tf.convert_to_tensor(sample['advantages'])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], tf.float32), (-1, self.n_agents, 1))
            batch_size = obs.shape[0]
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(batch_size, 1, 1))

            with tf.GradientTape() as tape:
                inputs = {'obs': obs, 'ids': IDs}
                _, a_dist, v_pred = self.policy(inputs)
                v_pred_tot = self.policy.value_tot(v_pred*agent_mask, state)
                v_true = rewards + self.gamma * returns
                log_prob = tf.reshape(a_dist.log_prob(actions), advantages.shape)
                entropy = tf.reshape(a_dist.entropy(), agent_mask.shape) * agent_mask

                loss_a = -tf.reduce_mean(advantages * log_prob * agent_mask)
                y_true = tf.reshape(tf.stop_gradient(v_true), [-1])
                y_pred = tf.reshape(v_pred_tot, [-1])
                loss_c = tk.losses.mean_squared_error(y_pred, y_true)
                loss_e = tf.reduce_mean(entropy)

                loss = loss_a + self.args.vf_coef * loss_c - self.args.ent_coef * loss_e
                gradients = tape.gradient(loss, self.policy.parameters_train)
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.args.clip_grad), var)
                    for (grad, var) in zip(gradients, self.policy.parameters_train)
                    if grad is not None
                ])

            # Logger
            lr = self.optimizer._decayed_lr(tf.float32)

            info = {
                "learning_rate": lr.numpy(),
                "actor_loss": loss_a.numpy(),
                "critic_loss": loss_c.numpy(),
                "entropy": loss_e.numpy(),
                "loss": loss.numpy(),
                "predict_value": tf.math.reduce_mean(v_pred).numpy()
            }

            return info
