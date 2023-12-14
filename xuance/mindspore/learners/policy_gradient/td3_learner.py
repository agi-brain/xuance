# TD3 add three tricks to DDPG:
# 1. noisy action in target actor
# 2. double critic network
# 3. delayed actor update
from xuance.mindspore.learners import *


class TD3_Learner(Learner):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(TD3_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._mean = ms.ops.ReduceMean(keep_dims=True)

        def construct(self, x):
            _, policy_q = self._backbone.Qpolicy(x)
            loss_p = -self._mean(policy_q)
            return loss_p

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone, gamma):
            super(TD3_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()
            self._gamma = gamma

        def construct(self, x, a, x_, r, d):
            _, action_q = self._backbone.Qaction(x, a)
            _, target_q = self._backbone.Qtarget(x_)
            backup = r + self._gamma * (1 - d) * target_q
            loss_q = self._loss(logits=action_q, labels=backup)
            return loss_q

    def __init__(self,
                 policy: nn.Cell,
                 optimizers: nn.Optimizer,
                 schedulers: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 delay: int = 3):
        self.tau = tau
        self.gamma = gamma
        self.delay = delay
        super(TD3_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
        self._expand_dims = ms.ops.ExpandDims()
        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy, self.gamma)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
        self.critic_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        info = {}
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        rew_batch = self._expand_dims(Tensor(rew_batch), 1)
        next_batch = Tensor(next_batch)
        ter_batch = self._expand_dims(Tensor(terminal_batch), 1)

        q_loss = self.critic_train(obs_batch, act_batch, next_batch, rew_batch, ter_batch)

        # actor update
        if self.iterations % self.delay == 0:
            p_loss = self.actor_train(obs_batch)
            self.policy.soft_update(self.tau)
            info["Ploss"] = p_loss.asnumpy()

        actor_lr = self.scheduler['actor'](self.iterations).asnumpy()
        critic_lr = self.scheduler['critic'](self.iterations).asnumpy()

        info.update({
            "Qloss": q_loss.asnumpy(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        })

        return info
