"""
Deep Q-Network (DQN)
Paper link: https://www.nature.com/articles/nature14236
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class DQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(DQN_Learner, self).__init__(config, policy, callback)
        self.optimizer = optim.Adam(params=self.policy.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.gather = ops.Gather(batch_dims=-1)
        self.n_actions = int(self.policy.action_dim)
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.policy.set_train()

    def forward_fn(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(axis=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        predictQ = self.gather(evalQ, act_batch, axis=-1).reshape(-1)
        loss = self.mse_loss(logits=predictQ, labels=ops.stop_gradient(targetQ))

        return loss, evalQ, targetQ, predictQ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'].reshape(-1, 1), dtype=ms.int32)
        rew_batch = Tensor(samples['rewards'])
        next_batch = Tensor(samples['obs_next'])
        ter_batch = Tensor(samples['terminals'])

        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        (loss, evalQ, targetQ, predictQ), grads = self.grad_fn(obs_batch, act_batch, next_batch, rew_batch, ter_batch)
        self.optimizer(grads)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "Qloss": loss.asnumpy(),
            "predictQ": predictQ.mean().asnumpy(),
            "learning_rate": lr.asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                policy=self.policy, info=info,
                                                evalQ=evalQ, predictQ=predictQ, targetQ=targetQ, loss=loss))

        return info
