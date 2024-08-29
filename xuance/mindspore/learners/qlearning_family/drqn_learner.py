"""
Deep Recurrent Q-Netwrk (DRQN)
Paper link: https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf
Implementation: MindSpore
"""
from xuance.mindspore import ms, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from argparse import Namespace
from mindspore.ops import OneHot
from mindspore.nn import MSELoss


class DRQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(DRQN_Learner, self).__init__(config, policy)
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

    def forward_fn(self, batch_size, x, a, label):
        rnn_hidden = self.policy.init_hidden(batch_size)
        _, _, _evalQ, _ = self.policy(x[:, 0:-1], *rnn_hidden)
        _predict_Q = (_evalQ * self.one_hot(a.astype(ms.int32), _evalQ.shape[-1],
                                            Tensor(1.0), Tensor(0.0))).sum(axis=-1)
        loss = self.mse_loss(_predict_Q, label)
        return loss, _evalQ

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        rew_batch = Tensor(samples['rewards'])
        ter_batch = Tensor(samples['terminals'])
        batch_size = obs_batch.shape[0]

        target_rnn_hidden = self.policy.init_hidden(batch_size)
        _, targetA, targetQ, _ = self.policy.target(obs_batch[:, 1:], *target_rnn_hidden)
        # targetQ = targetQ.max(dim=-1).values

        targetA = self.one_hot(targetA.astype(ms.int32), targetQ.shape[-1], Tensor(1.0), Tensor(0.0))
        targetQ = (targetQ * targetA).sum(axis=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        (loss, predictQ), grads = self.grad_fn(batch_size, obs_batch, act_batch, targetQ)
        self.optimizer(grads)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info = {
            "Qloss": loss.asnumpy(),
            "predictQ": predictQ.mean().asnumpy(),
            "learning_rate": lr.asnumpy(),
        }

        return info
