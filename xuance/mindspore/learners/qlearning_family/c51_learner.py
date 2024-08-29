"""
Distributional Reinforcement Learning (C51DQN)
Paper link: http://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace
from mindspore.ops import OneHot, Log, BatchMatMul, ExpandDims, Squeeze, ReduceSum, Abs, ReduceMean, clip_by_value
from mindspore.nn import MSELoss


class C51_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(C51_Learner, self).__init__(config, policy)
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
        self._log = Log()
        self._bmm = BatchMatMul()
        self._unsqueeze = ExpandDims()
        self._squeeze = Squeeze(1)
        self._sum = ReduceSum()
        self._mean = ReduceMean()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.clamp_min_value = Tensor(0.0, ms.float32)
        self.clamp_max_value = Tensor(1.0, ms.float32)
        self._abs = Abs()
        self._unsqueeze = ExpandDims()

    def forward_fn(self, x, a, projection, target_a, target_z):
        _, _, evalZ = self.policy(x)

        current_dist = self._sum(evalZ * self._unsqueeze(self.one_hot(a.astype(ms.int32), evalZ.shape[1],
                                                                      self.on_value, self.off_value), -1), 1)
        target_dist = self._sum(target_z * self._unsqueeze(self.one_hot(target_a.astype(ms.int32), evalZ.shape[1],
                                                                        self.on_value, self.off_value), -1), 1)

        target_dist = self._squeeze(self._bmm(self._unsqueeze(target_dist, 1),
                                              clip_by_value(projection, self.clamp_min_value, self.clamp_max_value)))
        loss = -self._mean(self._sum((target_dist * self._log(current_dist + 1e-8)), 1))

        return loss, evalZ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        _, targetA, targetZ = self.policy(next_batch)

        current_supports = self.policy.supports
        next_supports = self._unsqueeze(rew_batch, 1) + self.gamma * self.policy.supports * (
                    1 - self._unsqueeze(ter_batch, -1))
        next_supports = clip_by_value(next_supports, Tensor(self.policy.v_min, ms.float32),
                                      Tensor(self.policy.v_max, ms.float32))
        projection = 1 - self._abs(
            (self._unsqueeze(next_supports, -1) - self._unsqueeze(current_supports, 0))) / self.policy.deltaz

        (loss, evalZ), grads = self.grad_fn(obs_batch, act_batch, projection, targetA, targetZ)
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
