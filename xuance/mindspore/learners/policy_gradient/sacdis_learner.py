from xuance.mindspore.learners import *


class SACDIS_Learner(Learner):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(SACDIS_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone

        def construct(self, x):
            action_prob, log_pi, policy_q = self._backbone.Qpolicy(x)
            inside_term = 0.01 * log_pi - policy_q
            p_loss = (action_prob * inside_term).sum(axis=1).mean()
            return p_loss

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(SACDIS_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()

        def construct(self, x, a, backup):
            _, action_q = self._backbone.Qaction(x)
            # action_q = ms.ops.gather_elements(action_q, 1, (a.ceil()-1).astype(ms.int32))
            action_q = GatherD()(action_q, -1, a.astype(ms.int32))
            loss_q = self._loss(logits=action_q, labels=backup)
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
        super(SACDIS_Learner, self).__init__(policy, optimizers, schedulers, model_dir)
        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, optimizers['actor'])
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, optimizers['critic'])
        self.critic_train.set_train()
        self._unsqueeze = ms.ops.ExpandDims()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch).view(-1, 1)
        act_batch = self._unsqueeze(act_batch, -1)

        action_prob_next, log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_q = action_prob_next * (target_q - 0.01 * log_pi_next)
        target_q = self._unsqueeze(target_q.sum(axis=1), -1)
        rew = self._unsqueeze(rew_batch, -1)
        backup = rew + (1 - ter_batch) * self.gamma * target_q

        q_loss = self.critic_train(obs_batch, act_batch, backup)
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
