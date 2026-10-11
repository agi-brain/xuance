"""
Deep Deterministic Policy Gradient (DDPG)
Paper link: https://arxiv.org/pdf/1509.02971.pdf
Implementation: MindSpore
"""
from argparse import Namespace

from xuance.mindspore import ms, ops, nn, Module, Tensor, optim
from xuance.mindspore.learners import Learner
from xuance.mindspore.utils import clip_grads


class DDPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(DDPG_Learner, self).__init__(config, model, callback)
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
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        # Get gradient function
        self.grad_fn_actor = ms.value_and_grad(self.forward_fn_actor, None, self.optimizer['actor'].parameters,
                                               has_aux=True)
        self.grad_fn_critic = ms.value_and_grad(self.forward_fn_critic, None, self.optimizer['critic'].parameters,
                                                has_aux=True)
        self.model.set_train()
        self.grad_reducer_actor = self.get_grad_reducer(self.optimizer["actor"])
        self.grad_reducer_critic = self.get_grad_reducer(self.optimizer["critic"])

    @ms.jit
    def training_step_actor(self, obs_batch):
        (p_loss, policy_q), grads = self.grad_fn_actor(obs_batch)
        if self.distributed_training:
            grads = self.grad_reducer_actor(grads)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))

        self.optimizer["actor"](grads)

        return p_loss, policy_q

    @ms.jit
    def training_step_critic(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        (q_loss, action_q, next_q, target_q), grads = self.grad_fn_critic(
            obs_batch, act_batch, next_batch, rew_batch, ter_batch)
        if self.distributed_training:
            grads = self.grad_reducer_critic(grads)

        if self.use_grad_clip:
            grads = clip_grads(grads, Tensor(-self.grad_clip_norm), Tensor(self.grad_clip_norm))

        self.optimizer["critic"](grads)

        return q_loss, action_q, next_q, target_q

    def forward_fn_actor(self, obs_batch):
        policy_q = self.model.Qpolicy(obs_batch)
        p_loss = -ops.mean(policy_q)
        return p_loss, policy_q

    def forward_fn_critic(self, obs_batch, act_batch, next_batch, rew_batch, ter_batch):
        action_q = self.model.Qaction(obs_batch, act_batch).reshape([-1])
        next_q = self.model.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + (1 - ter_batch) * self.gamma * next_q
        q_loss = self.mse_loss(logits=action_q, labels=ops.stop_gradient(target_q))
        return q_loss, action_q, next_q, target_q

    def update(self, **samples):
        self.iterations += 1
        obs_batch = Tensor(samples['obs'], dtype=ms.float32)
        act_batch = Tensor(samples['actions'], dtype=ms.float32)
        rew_batch = Tensor(samples['rewards'], dtype=ms.float32)
        next_batch = Tensor(samples['obs_next'], dtype=ms.float32)
        ter_batch = Tensor(samples['terminals'], dtype=ms.float32)
        info = self.callback.on_update_start(self.iterations,
                                             model=self.model, obs=obs_batch, act=act_batch,
                                             next_obs=next_batch, rew=rew_batch, termination=ter_batch)

        q_loss, action_q, next_q, target_q = self.training_step_critic(
            obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        p_loss, policy_q = self.training_step_actor(obs_batch)

        self.model.soft_update(self.tau)

        self.scheduler['actor'].step()
        self.scheduler['critic'].step()
        actor_lr = self.scheduler['actor'].get_last_lr()[0]
        critic_lr = self.scheduler['critic'].get_last_lr()[0]

        info.update({
            "Qloss": q_loss.asnumpy(),
            "Ploss": p_loss.asnumpy(),
            "Qvalue": action_q.mean().asnumpy(),
            "actor_lr": actor_lr.asnumpy(),
            "critic_lr": critic_lr.asnumpy()
        })

        info.update(self.callback.on_update_end(self.iterations,
                                                model=self.model, info=info,
                                                action_q=action_q, next_q=next_q, target_q=target_q, policy_q=policy_q,
                                                q_loss=q_loss, p_loss=p_loss))

        return info
