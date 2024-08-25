"""
DQN with Double Q-learning (Double DQN)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/10295
Implementation: MindSpore
"""
import mindspore as ms
from xuance.mindspore import Module
from xuance.mindspore.learners import Learner
from argparse import Namespace
from mindspore.ops import OneHot


class DDQN_Learner(Learner):
    class PolicyNetWithLossCell(Module):
        def __init__(self, backbone, loss_fn):
            super(DDQN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self._onehot = OneHot()

        def construct(self, x, a, label):
            _, _, _evalQ = self._backbone(x)
            _predict_Q = (_evalQ * self._onehot(a.astype(ms.int32), _evalQ.shape[1], Tensor(1.0), Tensor(0.0))).sum(
                axis=-1)
            loss = self._loss_fn(_predict_Q, label)
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: Module):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.one_hot = OneHot()
        super(DDQN_Learner, self).__init__(config, policy)
        # define mindspore trainer
        loss_fn = nn.MSELoss()
        self.loss_net = self.PolicyNetWithLossCell(self.policy, loss_fn)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        rew_batch = Tensor(rew_batch)
        next_batch = Tensor(next_batch)
        ter_batch = Tensor(terminal_batch)

        _, targetA, targetQ = self.policy.target(next_batch)

        targetA = self.one_hot(targetA, targetQ.shape[1], Tensor(1.0), Tensor(0.0))
        targetQ = (targetQ * targetA).sum(axis=-1)
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
