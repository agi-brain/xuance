import numpy as np
from argparse import Namespace

from xuance.mindspore import ms, ops, nn, msd, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class NPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(NPG_Learner, self).__init__(config, model, callback)
        self.optimizer = {
            'actor': optim.Adam(params=self.model.actor.trainable_params(), lr=self.config.learning_rate, eps=1e-5),
            'critic': optim.Adam(params=self.model.critic.trainable_params(), lr=self.config.learning_rate, eps=1e-5),
        }
        self.scheduler = {
            'actor': optim.lr_scheduler.LinearLR(self.optimizer['actor'], start_factor=1.0,
                                                 end_factor=self.end_factor_lr_decay,
                                                 total_iters=self.config.running_steps),
            'critic': optim.lr_scheduler.LinearLR(self.optimizer['critic'], start_factor=1.0,
                                                  end_factor=self.end_factor_lr_decay,
                                                  total_iters=self.config.running_steps)
        }
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(axis=-1)
        self.is_continuous = self.model.is_continuous
        self.a_dist = msd.Normal(dtype=ms.float32) if self.is_continuous else msd.Categorical()
        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)
        self.model.set_train()
        self.grad_reducer_actor = self.get_grad_reducer(self.optimizer["actor"])
        self.grad_reducer_critic = self.get_grad_reducer(self.optimizer["critic"])


    # @ms.jit
    def training_step_actor(self, obs_batch, act_batch, adv_batch):
        (a_loss, log_prob), grads = self.grad_fn_actor(obs_batch, act_batch, adv_batch)

        # Natural policy gradient.
        for i, param in enumerate(self.model.actor.trainable_params()):
            fisher_info = self.compute_fisher_information(param, obs_batch, act_batch)
            grad = grads[i].reshape([-1])
            natural_grad = ops.matmul(fisher_info, grad)
            natural_grad = natural_grad.reshape(param.shape)
            grads[i].assign_value(natural_grad)

        if self.distributed_training:
            grads = self.grad_reducer_actor(grads)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))

        self.optimizer["actor"](grads)

        return a_loss, log_prob

    @ms.jit
    def training_step_critic(self, obs_batch, ret_batch):
        (v_pred, c_loss), grads = self.grad_fn_critic(obs_batch, ret_batch)
        if self.distributed_training:
            grads = self.grad_reducer_critic(grads)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))

        self.optimizer["critic"](grads)

        return v_pred, c_loss

    def forward_fn_actor(self, obs_batch, act_batch, adv_batch):
        if self.is_continuous:
            mu, std, _ = self.model.actor(obs_batch)
            log_prob = self.a_dist._log_prob(value=act_batch, mean=mu, sd=std)
            log_prob = ops.reduce_sum(x=log_prob, axis=-1)
        else:
            logits, _ = self.model.actor(obs_batch)
            probs = self.softmax(logits)
            log_prob = self.a_dist._log_prob(value=act_batch, probs=probs)

        a_loss = -ops.mean(adv_batch * log_prob)
        return a_loss, log_prob

    def forward_fn_critic(self, obs_batch, ret_batch):
        v_pred, _ = self.model.critic(obs_batch)
        c_loss = self.mse_loss(logits=v_pred, labels=ret_batch)
        return v_pred, c_loss

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'])
        act_batch = Tensor(samples['actions'])
        ret_batch = Tensor(samples['returns'])
        adv_batch = Tensor(samples['advantages'])
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             returns=ret_batch, advantages=adv_batch)

        a_loss, log_prob = self.training_step_actor(obs_batch, act_batch, adv_batch)

        v_pred, c_loss = self.training_step_critic(obs_batch, ret_batch)

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        lr_actor = self.scheduler['actor'].get_last_lr()[0]
        lr_critic = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "actor-loss": a_loss.asnumpy(),
            "critic-loss": c_loss.asnumpy(),
            "learning_rate_actor": lr_actor.asnumpy(),
            "learning_rate_critic": lr_critic.asnumpy(),
            "predict_value": v_pred.mean().asnumpy()
        })
        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                v_pred=v_pred, log_prob=log_prob,
                                                a_loss=a_loss, c_loss=c_loss))
        return info

    def compute_fisher_information(self, params, obs, act):
        param_num = 0
        for param in params:
            param_num += param.numel()
        fisher_information = ops.zeros(size=(param_num, param_num), dtype=params.dtype)

        def forward_fn(observations, actions):
            if self.is_continuous:
                mu, std, _ = self.model.actor(observations)
                log_prob = self.a_dist._log_prob(value=actions, mean=mu, sd=std)
                log_prob = ops.reduce_sum(x=log_prob, axis=-1)
            else:
                logits, _ = self.model.actor(observations)
                probs = self.softmax(logits)
                log_prob = self.a_dist._log_prob(value=actions, probs=probs)

            return log_prob.sum()

        grad_fn = ops.GradOperation(get_by_list=True)(forward_fn, params)
        grads = grad_fn(obs, act)
        score = grads.reshape([-1])
        fisher_information += ops.outer(score, score) * forward_fn(obs, act)
        fisher_information /= self.config.horizon_size
        fisher_information = fisher_information + 1e-3 * ops.eye(param_num, param_num, fisher_information.dtype)
        if self.device.upper() is "CPU" or "GPU":
            fisher_inv = np.linalg.inv(fisher_information.asnumpy())
            fisher_inv = Tensor(fisher_inv)
        else:
            fisher_inv = ms.mint.linalg.inv(fisher_information)
        return fisher_inv
