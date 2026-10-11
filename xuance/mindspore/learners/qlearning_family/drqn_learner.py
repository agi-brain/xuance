"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: MindSpore
"""
from argparse import Namespace

from xuance.mindspore import ms, nn, ops, Module, Tensor, optim, mutable
from xuance.mindspore.learners import Learner


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(DRQN_Learner, self).__init__(config, model, callback)
        self.optimizer = optim.Adam(params=self.model.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                     start_factor=1.0,
                                                     end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)

        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.gather = ops.Gather(batch_dims=-1)
        self.n_actions = self.model.n_actions
        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()
        self.grad_reducer = self.get_grad_reducer(self.optimizer)

    def forward_fn(self, batch_size, obs_batch, act_batch, rew_batch, ter_batch):
        rnn_states = self.model.init_rnn_states(batch_size)
        _, evalQ, _ = self.model(
            obs_batch[:, 0:-1],
            rnn_states=mutable(rnn_states) if type(rnn_states) is tuple else rnn_states
        )
        predictQ = self.gather(evalQ, act_batch.unsqueeze(-1), axis=-1).squeeze(-1)

        _, targetQ, targetA = self.model.target(
            obs_batch,
            rnn_states=mutable(rnn_states) if type(rnn_states) is tuple else rnn_states
        )
        targetQ = targetQ[:, 1:]
        targetA = targetA[:, 1:]
        # targetQ = targetQ.max(dim=-1).values

        targetQ = self.gather(targetQ, targetA.unsqueeze(-1), axis=-1).squeeze(-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        loss = self.mse_loss(logits=predictQ, labels=ops.stop_gradient(targetQ))
        return loss, evalQ, predictQ, targetA, targetQ

    @ms.jit
    def training_step(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        (loss, evalQ, predictQ, targetA, targetQ), grads = self.grad_fn(
            obs_batch, act_batch, next_batch, rew_batch, ter_batch
        )
        if self.distributed_training:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)

        return loss, evalQ, predictQ, targetA, targetQ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'], dtype=ms.int32)
        rew_batch = Tensor(samples['rewards'])
        ter_batch = Tensor(samples['terminals'])
        batch_size = obs_batch.shape[0]

        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             rew=rew_batch, termination=ter_batch, batch_size=batch_size)

        loss, evalQ, predictQ, targetA, targetQ = self.training_step(
            batch_size, obs_batch, act_batch, rew_batch, ter_batch
        )

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.model.copy_target()

        self.scheduler.step()

        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "Qloss": loss.asnumpy(),
            "predictQ": predictQ.mean().asnumpy(),
            "learning_rate": lr.asnumpy(),
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                evalQ=evalQ, predictQ=predictQ, targetQ=targetQ, loss=loss))

        return info
