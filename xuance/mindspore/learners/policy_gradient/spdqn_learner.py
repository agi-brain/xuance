from xuance.mindspore.learners import *
from mindspore.ops import OneHot


class SPDQN_Learner(Learner):
    class QNetWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(SPDQN_Learner.QNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, x, dis_a, con_a, label, input_q):
            # optimize q-network
            eval_qs = self._backbone.Qeval(x, con_a, input_q)
            eval_q = eval_qs.gather(dis_a.astype(ms.int32).view(-1, 1), 1).squeeze()
            q_loss = self._loss_fn(eval_q, label)
            return q_loss

    class ConActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(SPDQN_Learner.ConActorNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone

        def construct(self, x, input_q2):
            # optimize actor network
            policy_q = self._backbone.Qpolicy(x, input_q2)
            p_loss = - policy_q.mean()
            return p_loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Optional[Sequence[nn.exponential_decay_lr]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01):
        self.gamma = gamma
        self.tau = tau
        super(SPDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        # define loss function
        loss_fn = nn.MSELoss()
        # connect the feed forward network with loss function.
        self.q_loss_net = self.QNetWithLossCell(policy, loss_fn)
        self.con_loss_net = self.ConActorNetWithLossCell(policy)
        # define the training network
        self.con_actor_train = nn.TrainOneStepCell(self.con_loss_net, optimizer[0])
        self.q_net_train = nn.TrainOneStepCell(self.q_loss_net, optimizer[1])
        # set the training network as train mode.
        self.con_actor_train.set_train()
        self.q_net_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        hyact_batch = Tensor(act_batch)
        disact_batch = hyact_batch[:, 0]  # .long()
        conact_batch = hyact_batch[:, 1:]
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        target_conact = self.policy.Atarget(next_batch)
        target_q = self.policy.Qtarget(next_batch, target_conact)
        target_q = target_q.max(axis=-1)
        target_q = rew_batch + (1 - ter_batch) * self.gamma * target_q

        batch_size = obs_batch.shape[0]
        input_q = self.policy._concat((obs_batch, self.policy._zeroslike(conact_batch)))
        input_q = input_q.repeat(self.policy.num_disact, 0)
        input_q = input_q.asnumpy()
        conact_batch = conact_batch.asnumpy()
        for i in range(self.policy.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.policy.obs_size + self.policy.offsets[i]: self.policy.obs_size + self.policy.offsets[i + 1]] \
                = conact_batch[:, self.policy.offsets[i]:self.policy.offsets[i + 1]]
        input_q = ms.Tensor(input_q, dtype=ms.float32)
        conact_batch = Tensor(conact_batch)

        conact = self.policy.conactor(obs_batch)
        input_q2 = self.policy._concat((obs_batch, self.policy._zeroslike(conact)))
        input_q2 = input_q2.repeat(self.policy.num_disact, 0)
        input_q2 = input_q2.asnumpy()
        conact = conact.asnumpy()
        for i in range(self.policy.num_disact):
            input_q2[i * batch_size:(i + 1) * batch_size,
            self.policy.obs_size + self.policy.offsets[i]: self.policy.obs_size + self.policy.offsets[i + 1]] \
                = conact[:, self.policy.offsets[i]:self.policy.offsets[i + 1]]
        input_q2 = ms.Tensor(input_q2, dtype=ms.float32)

        q_loss = self.q_net_train(obs_batch, disact_batch, conact_batch, target_q, input_q)
        p_loss = self.con_actor_train(obs_batch, input_q2)

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
