from xuance.mindspore.learners import *
from mindspore.nn.probability.distribution import Normal


class SAC_Learner(Learner):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(SAC_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone

        def construct(self, x):
            _, log_pi, policy_q = self._backbone.Qpolicy(x)
            loss_a = (0.01 * log_pi.reshape([-1]) - policy_q).mean()
            return loss_a

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(SAC_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()

        def construct(self, x, a, target):
            _, action_q = self._backbone.Qaction(x, a)
            loss_q = self._loss(logits=action_q, labels=target)
            return loss_q

    def __init__(self,
                 policy: nn.Cell,
                 optimizers: nn.Optimizer,
                 schedulers: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.tau = tau
        self.gamma = gamma
        super(SAC_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
        self.critic_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target = rew_batch + (1 - ter_batch) * self.gamma * (target_q - 0.01 * log_pi_next.reshape([-1]))

        q_loss = self.critic_train(obs_batch, act_batch, target)
        p_loss = self.actor_train(obs_batch)

        self.policy.soft_update(self.tau)

        actor_lr = self.scheduler['actor'](self.iterations).asnumpy()
        critic_lr = self.scheduler['critic'](self.iterations).asnumpy()

        info = {
            "Qloss": q_loss.asnumpy(),
            "Ploss": p_loss.asnumpy(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }

        return info
