"""
Advantage Actor-Critic (A2C)
Implementation: MindSpore
"""
from argparse import Namespace
from xuance.mindspore import ms, nn, msd, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(A2C_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                     start_factor=1.0,
                                                     end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.policy.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, obs_batch, act_batch, adv_batch, ret_batch):
        if self.is_continuous:
            outputs, mu, std, v_pred = self.policy(obs_batch)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = ops.reduce_sum(x=log_prob, axis=-1)
            entropy = self.a_dist._entropy(mean=mu, sd=std)
            entropy = ops.reduce_sum(x=entropy, axis=-1)
        else:
            outputs, logits, v_pred = self.policy(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)
            entropy = self.a_dist.entropy(probs=probs)

        a_loss = -ops.mean(adv_batch * log_prob)
        c_loss = self.mse_loss(logits=v_pred, labels=ops.stop_gradient(ret_batch))
        e_loss = ops.mean(entropy)

        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        return loss, a_loss, c_loss, e_loss, v_pred

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])

        (loss, a_loss, c_loss, e_loss, v_pred), grads = self.grad_fn(obs_batch, act_batch, adv_batch, ret_batch)
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "total-loss": loss.asnumpy(),
            "actor-loss": a_loss.asnumpy(),
            "critic-loss": c_loss.asnumpy(),
            "entropy": e_loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "predict_value": v_pred.mean().asnumpy(),
        }

        return info
