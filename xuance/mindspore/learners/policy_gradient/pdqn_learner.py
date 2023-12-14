from xuance.mindspore.learners import *
from mindspore.ops import OneHot


class PDQN_Learner(Learner):
    class ConActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(PDQN_Learner.ConActorNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone

        def construct(self, x):
            # optimize actor network
            policy_q = self._backbone.Qpolicy(x)
            p_loss = - policy_q.mean()
            return p_loss
    
    class QNetWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(PDQN_Learner.QNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, x, dis_a, con_a, label):
            # optimize q-network
            eval_qs = self._backbone.Qeval(x, con_a)
            eval_q = eval_qs.gather(dis_a.astype(ms.int32).view(-1, 1), 1).squeeze()
            q_loss = self._loss_fn(eval_q, label)
            return q_loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Optional[Sequence[nn.exponential_decay_lr]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.gamma = gamma
        self.tau = tau
        super(PDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        # define loss function
        loss_fn = nn.MSELoss()
        # connect the feed forward network with loss function.
        self.con_loss_net = self.ConActorNetWithLossCell(policy)
        self.q_loss_net = self.QNetWithLossCell(policy, loss_fn)
        # define the training network
        self.con_policy_train = nn.TrainOneStepCell(self.con_loss_net, optimizer[0])
        self.q_policy_train = nn.TrainOneStepCell(self.q_loss_net, optimizer[1])
        # set the training network as train mode.
        self.con_policy_train.set_train()
        self.q_policy_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        hyact_batch = Tensor(act_batch)
        disact_batch = hyact_batch[:, 0]#.long()
        conact_batch = hyact_batch[:, 1:]
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        target_conact = self.policy.Atarget(next_batch)
        target_q = self.policy.Qtarget(next_batch, target_conact)
        target_q = target_q.max(axis=-1)
        target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

        q_loss = self.q_policy_train(obs_batch, disact_batch, conact_batch, target_q)
        p_loss = self.con_policy_train(obs_batch)

        self.policy.soft_update(self.tau)

        con_actor_lr = self.scheduler[0](self.iterations).asnumpy()
        qnet_lr = self.scheduler[1](self.iterations).asnumpy()

        info = {
            "P_loss": p_loss.asnumpy(),
            "Q_loss": q_loss.asnumpy(),
            "con_actor_lr": con_actor_lr,
            "qnet_lr": qnet_lr
        }

        return info
