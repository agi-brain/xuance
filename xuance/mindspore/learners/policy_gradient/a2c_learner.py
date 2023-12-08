from xuance.mindspore.learners import *


class A2C_Learner(Learner):
    class ACNetWithLossCell(nn.Cell):
        def __init__(self, backbone, ent_coef, vf_coef):
            super(A2C_Learner.ACNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._mean = ms.ops.ReduceMean(keep_dims=True)
            self._loss_c = nn.MSELoss()
            self._ent_coef = ent_coef
            self._vf_coef = vf_coef

        def construct(self, x, a, adv, r):
            _, act_probs, v_pred = self._backbone(x)
            log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs)
            loss_a = -self._mean(adv * log_prob)
            loss_c = self._loss_c(logits=v_pred, labels=r)
            loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
            loss = loss_a - self._ent_coef * loss_e + self._vf_coef * loss_c

            return loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_grad: Optional[float] = None,
                 clip_type: Optional[int] = None):
        super(A2C_Learner, self).__init__(policy, optimizer, scheduler, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_grad = clip_grad
        # define mindspore trainer
        self.loss_net = self.ACNetWithLossCell(policy, self.ent_coef, self.vf_coef)
        # self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, optimizer,
                                                         clip_type=clip_type, clip_value=clip_grad)
        self.policy_train.set_train()

    def update(self, obs_batch, act_batch, ret_batch, adv_batch):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        ret_batch = Tensor(ret_batch)
        adv_batch = Tensor(adv_batch)

        loss = self.policy_train(obs_batch, act_batch, adv_batch, ret_batch)

        # Logger
        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "total-loss": loss.asnumpy(),
            "learning_rate": lr
        }

        return info
