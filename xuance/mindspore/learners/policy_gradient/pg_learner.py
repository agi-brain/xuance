"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: MindSpore
"""
import mindspore as ms
from xuance.mindspore import Module
from xuance.mindspore.learners import Learner
from argparse import Namespace


class PG_Learner(Learner):
    class PolicyNetWithLossCell(Module):
        def __init__(self, backbone, ent_coef):
            super(PG_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._ent_coef = ent_coef
            self._mean = ms.ops.ReduceMean(keep_dims=True)

        def construct(self, x, a, r):
            _, act_probs = self._backbone(x)
            log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs)
            loss_a = -self._mean(r * log_prob)
            loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
            loss = loss_a - self._ent_coef * loss_e
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PG_Learner, self).__init__(config, policy)
        self.optimizer = ms.nn.Adam(params=policy.trainable_params(), learning_rate=self.config.learning_rate, eps=1e-5)
        # define mindspore trainer
        self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef)
        # self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, optimizer,
                                                         clip_type=clip_type, clip_value=clip_grad)
        self.policy_train.set_train()

    def update(self, obs_batch, act_batch, ret_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        ret_batch = Tensor(ret_batch)

        loss = self.policy_train(obs_batch, act_batch, ret_batch)

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "total-loss": loss.asnumpy(),
            "learning_rate": lr
        }

        return info
