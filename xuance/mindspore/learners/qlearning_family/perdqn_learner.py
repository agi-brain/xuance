from xuance.mindspore.learners import *
from mindspore.ops import OneHot


class PerDQN_Learner(Learner):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(PerDQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self._onehot = OneHot()

        def construct(self, x, a, label):
            _, _, _evalQ, _ = self._backbone(x)
            _predict_Q = (_evalQ * self._onehot(a, _evalQ.shape[1], Tensor(1.0), Tensor(0.0))).sum(axis=-1)
            loss = self._loss_fn(_predict_Q, label)
            return loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(PerDQN_Learner, self).__init__(policy, optimizer, scheduler, summary_writer, modeldir)
        # define loss function
        loss_fn = nn.MSELoss()
        # connect the feed forward network with loss function.
        self.loss_net = self.PolicyNetWithLossCell(policy, loss_fn)
        # define the training network
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        # set the training network as train mode.
        self.policy_train.set_train()

        self._onehot = OneHot()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch, ms.int32)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, _, _, targetQ = self.policy(next_batch)
        targetQ = targetQ.max(axis=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        _, _, evalQ, _ = self.policy(obs_batch)
        predict_Q = (evalQ * self._onehot(act_batch, evalQ.shape[1], Tensor(1.0), Tensor(0.0))).sum(axis=-1)
        td_error = targetQ - predict_Q

        loss = self.policy_train(obs_batch, act_batch, targetQ)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()
        self.writer.add_scalar("Qloss", loss.asnumpy(), self.iterations)
        self.writer.add_scalar("learning_rate", lr, self.iterations)

        return np.abs(td_error.asnumpy())
