from xuance.mindspore.learners import *
from mindspore.ops import OneHot,ExpandDims,ReduceSum


class QRDQN_Learner(Learner):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(QRDQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self._onehot = OneHot()
            self.on_value = Tensor(1.0, ms.float32)
            self.off_value = Tensor(0.0, ms.float32)
            self._unsqueeze = ExpandDims()
            self._sum = ReduceSum()

        def construct(self, x, a, target_quantile):
            _,_,evalZ = self._backbone(x)
            current_quantile = self._sum(evalZ * self._unsqueeze(self._onehot(a, evalZ.shape[1], self.on_value, self.off_value), -1), 1)
            loss = self._loss_fn(target_quantile, current_quantile)
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
        super(QRDQN_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        # define loss function
        loss_fn = nn.MSELoss()
        # connect the feed forward network with loss function.
        self.loss_net = self.PolicyNetWithLossCell(policy, loss_fn)
        # define the training network
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        # set the training network as train mode.
        self.policy_train.set_train()

        self._onehot = OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self._unsqueeze = ExpandDims()
        self._sum = ReduceSum()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch, ms.int32)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, targetA, targetZ = self.policy(next_batch)
        target_quantile = self._sum(targetZ * self._unsqueeze(self._onehot(targetA, targetZ.shape[1], self.on_value, self.off_value), -1), 1)
        target_quantile = self._unsqueeze(rew_batch, 1) + self.gamma * target_quantile * (1-self._unsqueeze(ter_batch, 1))

        loss = self.policy_train(obs_batch, act_batch, target_quantile)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "Qloss": loss.asnumpy(),
            "learning_rate": lr
        }

        return info
