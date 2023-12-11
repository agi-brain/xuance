from xuance.mindspore.learners import *
from mindspore.ops import OneHot


class DQN_Learner(Learner):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(DQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self._onehot = OneHot()

        def construct(self, x, a, label):
            _, _, _evalQ = self._backbone(x)
            _predict_Q = (_evalQ * self._onehot(a.astype(ms.int32), _evalQ.shape[1], Tensor(1.0), Tensor(0.0))).sum(axis=-1)
            loss = self._loss_fn(_predict_Q, label)
            return loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(DQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        # define loss function
        loss_fn = nn.MSELoss()
        # connect the feed forward network with loss function.
        self.loss_net = self.PolicyNetWithLossCell(policy, loss_fn)
        # define the training network
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        # set the training network as train mode.
        self.policy_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(axis=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ

        loss = self.policy_train(obs_batch, act_batch, targetQ)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "Qloss": loss.asnumpy(),
            "learning_rate": lr
        }

        return info
