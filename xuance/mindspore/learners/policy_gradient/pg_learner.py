"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PG_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.ent_coef = config.ent_coef
        self._mean = ms.ops.ReduceMean(keep_dims=True)
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, x, a, r):
        _, a_dist, _ = self.policy(x)
        log_prob = a_dist.log_prob(a)
        loss_a = -self._mean(r * log_prob)
        loss_e = self._mean(a_dist.entropy())
        loss = loss_a - self.ent_coef * loss_e
        return loss, loss_a, loss_e

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])

        (loss, loss_a, loss_e), grads = self.grad_fn(obs_batch, act_batch, ret_batch)
        self.optimizer(grads)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "total-loss": loss.asnumpy(),
            "actor-loss": loss_a.asnumpy(),
            "entropy": loss_e.asnumpy(),
            "learning_rate": lr.asnumpy(),
        }

        return info
