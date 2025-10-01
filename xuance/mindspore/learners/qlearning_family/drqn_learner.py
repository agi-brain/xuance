"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        super(DRQN_Learner, self).__init__(config, policy, callback)
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

    def forward_fn(self, batch_size, obs_batch, act_batch, rew_batch, ter_batch):
        rnn_hidden = self.policy.init_hidden(batch_size)
        _, _, evalQ, _ = self.policy(obs_batch[:, 0:-1], *rnn_hidden)
        predictQ = self.gather(evalQ, act_batch.unsqueeze(-1), axis=-1).squeeze(-1)

        target_rnn_hidden = self.policy.init_hidden(batch_size)
        _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
        # targetQ = targetQ.max(dim=-1).values

        targetQ = self.gather(targetQ, targetA.unsqueeze(-1), axis=-1).squeeze(-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        loss = self.mse_loss(logits=predictQ, labels=ops.stop_gradient(targetQ))
        return loss, evalQ, predictQ, targetA, targetQ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'], dtype=ms.int32)
        rew_batch = Tensor(samples['rewards'])
        ter_batch = Tensor(samples['terminals'])
        batch_size = obs_batch.shape[0]

        info = self.callback.on_update_start(self.iterations,
                                             policy=self.policy, obs=obs_batch, act=act_batch,
                                             rew=rew_batch, termination=ter_batch, batch_size=batch_size)

        (loss, evalQ, predictQ, targetA, targetQ), grads = self.grad_fn(
            batch_size, obs_batch, act_batch, rew_batch, ter_batch)
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
                                                evalQ=evalQ, predictQ=predictQ, targetA=targetA, targetQ=targetQ,
                                                loss=loss))

        return info
