"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: MindSpore
"""
from argparse import Namespace
from mindspore import nn
from xuance.mindspore import ms, msd, ops, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(PG_Learner, self).__init__(config, model, callback)
        self.optimizer = optim.Adam(params=self.model.trainable_params(), lr=self.config.learning_rate, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay,
                                                     total_iters=self.config.running_steps)
        self.ent_coef = config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.model.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()

        # Get gradient function
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.model.set_train()
        self.grad_reducer = self.get_grad_reducer(self.optimizer)

    @ms.jit
    def training_step(self, obs_batch, act_batch, ret_batch):
        (loss, a_loss, e_loss, log_prob), grads = self.grad_fn(obs_batch, act_batch, ret_batch)
        if self.distributed_training:
            grads = self.grad_reducer(grads)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))

        self.optimizer(grads)

        return loss, a_loss, e_loss, log_prob

    def forward_fn(self, obs_batch, act_batch, ret_batch):
        if self.is_continuous:
            mu, log_std = self.model(obs_batch)
            std = ops.exp(log_std)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = ops.reduce_sum(x=log_prob, axis=-1)
            entropy = self.a_dist._entropy(mean=mu, sd=std)
            entropy = ops.reduce_sum(x=entropy, axis=-1)
        else:
            logits = self.model(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)
            entropy = self.a_dist.entropy(probs=probs)

        a_loss = -ops.mean(ret_batch * log_prob)
        e_loss = ops.mean(entropy)

        loss = a_loss - self.ent_coef * e_loss
        return loss, a_loss, e_loss, log_prob

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch, returns=ret_batch)

        loss, a_loss, e_loss, log_prob = self.training_step(obs_batch, act_batch, ret_batch)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        info.update({
            "total-loss": loss.asnumpy(),
            "actor-loss": a_loss.asnumpy(),
            "entropy": e_loss.asnumpy(),
            "learning_rate": lr.asnumpy(),
        })
        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                log_prob=log_prob, a_loss=a_loss, e_loss=e_loss, loss=loss))
        return info
