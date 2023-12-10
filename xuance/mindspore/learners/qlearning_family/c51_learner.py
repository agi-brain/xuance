from xuance.mindspore.learners import *
from mindspore.ops import OneHot,Log,BatchMatMul,ExpandDims,Squeeze,ReduceSum,Abs,ReduceMean,clip_by_value


class C51_Learner(Learner):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(C51_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._onehot = OneHot()
            self._log = Log()
            self._bmm = BatchMatMul()
            self._unsqueeze = ExpandDims()
            self._squeeze = Squeeze(1)
            self._sum = ReduceSum()
            self._mean = ReduceMean()
            self.on_value = Tensor(1.0, ms.float32)
            self.off_value = Tensor(0.0, ms.float32)
            self.clamp_min_value = Tensor(0.0, ms.float32)
            self.clamp_max_value = Tensor(1.0, ms.float32)

        def construct(self, x, a, projection, target_a, target_z):
            _, _, evalZ = self._backbone(x)
            
            current_dist = self._sum(evalZ * self._unsqueeze(self._onehot(a, evalZ.shape[1], self.on_value, self.off_value), -1), 1)
            target_dist = self._sum(target_z * self._unsqueeze(self._onehot(target_a, evalZ.shape[1], self.on_value, self.off_value), -1), 1)
            
            target_dist = self._squeeze(self._bmm(self._unsqueeze(target_dist, 1),clip_by_value(projection,self.clamp_min_value,self.clamp_max_value)))
            loss = -self._mean(self._sum((target_dist *  self._log(current_dist + 1e-8)), 1))

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
        super(C51_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        # connect the feed forward network with loss function.
        self.loss_net = self.PolicyNetWithLossCell(policy)
        # define the training network
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        # set the training network as train mode.
        self.policy_train.set_train()

        self._abs = Abs()
        self._unsqueeze = ExpandDims()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch, ms.int32)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, targetA, targetZ = self.policy(next_batch)
        
        current_supports = self.policy.supports
        next_supports = self._unsqueeze(rew_batch, 1) + self.gamma * self.policy.supports * (1-self._unsqueeze(ter_batch, -1))
        next_supports = clip_by_value(next_supports, Tensor(self.policy.vmin, ms.float32), Tensor(self.policy.vmax, ms.float32))
        projection = 1 - self._abs((self._unsqueeze(next_supports, -1) - self._unsqueeze(current_supports, 0)))/self.policy.deltaz

        loss = self.policy_train(obs_batch, act_batch, projection, targetA, targetZ)

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "Qloss": loss.asnumpy(),
            "learning_rate": lr
        }

        return info
