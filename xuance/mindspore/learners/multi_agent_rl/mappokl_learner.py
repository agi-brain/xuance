"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class MAPPO_KL_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents, vf_coef, ent_coef, clip_range, use_value_clip, value_clip_range, target_kl):
            super(MAPPO_KL_Learner.PolicyNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.clip_range = clip_range
            self.use_value_clip = use_value_clip
            self.value_clip_range = Tensor(value_clip_range)
            self.mse_loss = nn.MSELoss()
            self.exp = ops.Exp()
            self.log = ops.Log()
            self.miminum = ops.Minimum()
            self.maximum = ops.Maximum()
            self.expand_dims = ops.ExpandDims()
            self.broadcast_to = ops.BroadcastTo((-1, self.n_agents, -1))
            self.target_kl = Tensor(target_kl)
            self.kl_coef = ms.Parameter(Tensor(1.0))

        def construct(self, bs, s, o, a, act_prob_old, ret, adv, agt_mask, ids):
            _, act_prob = self._backbone(o, ids)
            log_pi = self._backbone.actor.log_prob(value=a, probs=act_prob)
            kl = self._backbone.actor.kl_div(act_prob, act_prob_old).mean()
            act_prob_old_select = GatherD()(act_prob_old, -1, ops.expand_dims(a, -1).astype(ms.int32))
            log_pi_old = self.log(act_prob_old_select.squeeze(-1))
            ratio = self.exp(log_pi - log_pi_old).view(bs, self.n_agents, 1)
            advantages_mask = adv * agt_mask
            loss_a = -(ratio * advantages_mask).mean() + self.kl_coef * kl

            entropy = self._backbone.actor.entropy(probs=act_prob).reshape(agt_mask.shape) * agt_mask
            loss_e = entropy.mean()

            state_expand = self.broadcast_to(self.expand_dims(s, -2))
            value = self._backbone.values(state_expand, ids) * agt_mask
            if self.use_value_clip:
                value_clipped = ret + ops.clip_by_value(value - ret, -self.value_clip_range, self.value_clip_range)
                value_target = advantages_mask + ret * agt_mask
                loss_v = (value - value_target) ** 2
                loss_v_clipped = (value_clipped * agt_mask - value_target) ** 2
                loss_c = self.maximum(loss_v, loss_v_clipped).mean()
            else:
                loss_c = self.mse_loss(value, ret * agt_mask)

            # if kl > self.target_kl * 1.5:
            #     self.kl_coef = self.kl_coef * 2.
            # elif kl < self.target_kl * 0.5:
            #     self.kl_coef = self.kl_coef / 2.
            # self.kl_coef = ops.clip_by_value(self.kl_coef, Tensor(0.1), Tensor(20))

            loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.value_clip_range = config.value_clip_range
        self.mse_loss = nn.MSELoss()
        super(MAPPO_KL_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        # define mindspore trainers
        self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, config.vf_coef, config.ent_coef,
                                                   config.clip_range, config.use_value_clip, config.value_clip_range,
                                                   config.target_kl)
        if self.args.use_grad_norm:
            self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer,
                                                             clip_type=config.clip_type, clip_value=config.clip_grad)
        else:
            self.policy_train = nn.TrainOneStepCell(self.loss_net, self.optimizer)

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        act_prob_old = Tensor(sample['act_prob_old'])
        returns = Tensor(sample['values'])
        advantages = Tensor(sample['advantages'])
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        loss = self.policy_train(batch_size, state, obs, actions, act_prob_old, returns, advantages, agent_mask, IDs)

        # Logger
        lr = self.scheduler(self.iterations).asnumpy()
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss", loss.asnumpy(), self.iterations)
