from xuance.mindspore.learners import *


class PPOCLIP_Learner(Learner):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, ent_coef, vf_coef, clip_range):
            super(PPOCLIP_Learner.PolicyNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._ent_coef = ent_coef
            self._vf_coef = vf_coef
            self._clip_range = [Tensor(1.0 - clip_range), Tensor(1.0 + clip_range)]
            self._exp = ms.ops.Exp()
            self._minimum = ms.ops.Minimum()
            self._mean = ms.ops.ReduceMean(keep_dims=True)
            self._loss = nn.MSELoss()

        def construct(self, x, a, old_log_p, adv, ret):
            outputs, act_probs, v_pred = self._backbone(x)
            log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs)
            ratio = self._exp(log_prob - old_log_p)
            surrogate1 = ms.ops.clip_by_value(ratio, self._clip_range[0], self._clip_range[1]) * adv
            surrogate2 = adv * ratio
            loss_a = -self._mean(self._minimum(surrogate1, surrogate2))
            loss_c = self._loss(logits=v_pred, labels=ret)
            loss_e = self._mean(self._backbone.actor.entropy(probs=act_probs))
            loss = loss_a - self._ent_coef * loss_e + self._vf_coef * loss_c
            return loss

    def __init__(self,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25):
        super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, summary_writer, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        # define mindspore trainer
        self.loss_net = self.PolicyNetWithLossCell(policy, self.ent_coef, self.vf_coef, self.clip_range)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def update(self, obs_batch, act_batch, ret_batch, adv_batch, old_logp):
        self.iterations += 1
        obs_batch = Tensor(obs_batch)
        act_batch = Tensor(act_batch)
        ret_batch = Tensor(ret_batch)
        adv_batch = Tensor(adv_batch)
        old_logp_batch = Tensor(old_logp)

        loss = self.policy_train(obs_batch, act_batch, old_logp_batch, adv_batch, ret_batch)
        # Logger
        lr = self.scheduler(self.iterations).asnumpy()
        self.writer.add_scalar("tot-loss", loss.asnumpy(), self.iterations)
        self.writer.add_scalar("learning_rate", lr, self.iterations)
