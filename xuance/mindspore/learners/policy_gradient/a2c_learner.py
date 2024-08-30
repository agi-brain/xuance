"""
Advantage Actor-Critic (A2C)
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace
from mindspore.nn import MSELoss


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(A2C_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.mse_loss = MSELoss()
        self._mean = ms.ops.ReduceMean(keep_dims=True)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, x, a, adv, r):
        _, a_dist, v_pred = self.policy(x)
        log_prob = a_dist.log_prob(a)
        loss_a = -self._mean(adv * log_prob)
        loss_c = self.mse_loss(logits=v_pred, labels=r)
        loss_e = self._mean(a_dist.entropy())
        loss = loss_a - self.ent_coef * loss_e + self.vf_coef * loss_c

        return loss, loss_a, loss_e, loss_c, v_pred

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])

        (loss, loss_a, loss_e, loss_c, v_pred), grads = self.grad_fn(obs_batch, act_batch, adv_batch, ret_batch)
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "total-loss": loss.asnumpy(),
            "actor-loss": loss_a.asnumpy(),
            "critic-loss": loss_c.asnumpy(),
            "entropy": loss_e.asnumpy(),
            "learning_rate": lr.asnumpy(),
            "predict_value": v_pred.mean().asnumpy(),
        }

        return info
