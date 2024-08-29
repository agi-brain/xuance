"""
DQN with Quantile Regression (QRDQN)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11791
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace
from mindspore.ops import OneHot, ExpandDims, ReduceSum
from mindspore.nn import MSELoss


class QRDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(QRDQN_Learner, self).__init__(config, policy)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                     total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = MSELoss()
        self.one_hot = OneHot()
        self.n_actions = self.policy.action_dim
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self._unsqueeze = ExpandDims()
        self._sum = ReduceSum()

    def forward_fn(self, x, a, target_quantile):
        _, _, evalZ = self.policy(x)
        current_quantile = self._sum(evalZ * self._unsqueeze(self.one_hot(a.astype(ms.int32), evalZ.shape[1],
                                                                          self.on_value, self.off_value), -1), 1)
        loss = self.mse_loss(target_quantile, current_quantile)
        return loss, evalZ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        _, targetA, targetZ = self.policy(next_batch)
        target_quantile = self._sum(targetZ * self._unsqueeze(self.one_hot(targetA.astype(ms.int32), targetZ.shape[1],
                                                                           self.on_value, self.off_value), -1), 1)
        target_quantile = self._unsqueeze(rew_batch, 1) + self.gamma * target_quantile * (
                    1 - self._unsqueeze(ter_batch, 1))

        (loss, evalZ), grads = self.grad_fn(obs_batch, act_batch, target_quantile)
        self.optimizer(grads)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "Qloss": loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
        }

        return info
