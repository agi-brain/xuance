"""
Distributional Reinforcement Learning (C51DQN)
Paper link: http://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class C51_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(C51_Learner, self).__init__(config, policy, callback)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.gather = ops.Gather(batch_dims=-1)
        self.n_actions = self.policy.action_dim
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        _, _, evalZ = self.policy(obs_batch)
        _, targetA, targetZ = self.policy.target(next_batch)

        current_dist = self.gather(evalZ, act_batch, axis=1).squeeze(1)
        target_dist = self.gather(targetZ, targetA.unsqueeze(-1), axis=1).squeeze(1)

        current_supports = self.policy.supports
        next_supports = rew_batch.unsqueeze(1) + self.gamma * self.policy.supports * (1 - ter_batch.unsqueeze(1))
        next_supports = ops.clamp(next_supports, self.policy.v_min, self.policy.v_max)

        projection = 1 - ops.abs((next_supports.unsqueeze(-1) - current_supports.unsqueeze(0))) / self.policy.deltaz
        target_dist = ops.bmm(target_dist.unsqueeze(1), ops.clamp(projection, 0, 1)).squeeze(1)
        target_dist = ops.stop_gradient(target_dist)

        loss = -ops.mean(ops.sum(target_dist * ops.log(current_dist + 1e-8), dim=1))

        return loss, evalZ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'].reshape(-1, 1), dtype=ms.int32)
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        (loss, evalZ), grads = self.grad_fn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)
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
