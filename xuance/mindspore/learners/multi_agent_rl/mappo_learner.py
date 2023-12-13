"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://arxiv.org/pdf/2103.01955.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *
from xuance.mindspore.utils.operations import update_linear_decay


class MAPPO_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents, vf_coef, ent_coef, clip_range, use_value_clip, value_clip_range,
                     use_huber_loss):
            super(MAPPO_Learner.PolicyNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.clip_range = clip_range * 0.5
            self.use_value_clip = use_value_clip
            self.value_clip_range = Tensor(value_clip_range)
            self.use_huber_loss = use_huber_loss
            self.mse_loss = nn.MSELoss()
            self.huber_loss = nn.HuberLoss()
            self.exp = ops.Exp()
            self.miminum = ops.Minimum()
            self.maximum = ops.Maximum()
            self.expand_dims = ops.ExpandDims()
            self.broadcast_to = ops.BroadcastTo((-1, self.n_agents, -1))

        def construct(self, bs, s, o, a, log_pi_old, values, returns, advantages, agt_mask, ids):
            # actor loss
            _, act_probs = self._backbone(o, ids)
            log_pi = self._backbone.actor.log_prob(value=a, probs=act_probs)
            ratio = self.exp(log_pi - log_pi_old).view(bs, self.n_agents, 1)
            advantages_mask = advantages * agt_mask
            surrogate1 = ratio * advantages_mask
            surrogate2 = ops.clip_by_value(ratio, Tensor(1 - self.clip_range), Tensor(1 + self.clip_range)) * advantages_mask
            loss_a = -self.miminum(surrogate1, surrogate2).sum(axis=-2, keepdims=True).mean()

            # entropy loss
            entropy = self._backbone.actor.entropy(probs=act_probs).reshape(agt_mask.shape) * agt_mask
            loss_e = entropy.mean()

            # critic loss
            critic_in = self.broadcast_to(o.reshape([bs, 1, -1]))
            _, value_pred = self._backbone.get_values(critic_in, ids)
            value_pred = value_pred * agt_mask
            value_target = returns
            if self.use_value_clip:
                value_clipped = values + ops.clip_by_value(value_pred - values, -self.value_clip_range, self.value_clip_range)
                if self.use_huber_loss:
                    loss_v = self.huber_loss(value_pred, value_target)
                    loss_v_clipped = self.huber_loss(value_clipped, value_target)
                else:
                    loss_v = (value_pred - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                loss_c = self.maximum(loss_v, loss_v_clipped) * agt_mask
                loss_c = loss_c.sum() / agt_mask.sum()
            else:
                if self.use_huber_loss:
                    loss_v = self.huber_loss(logits=value_pred, labels=value_target) * agt_mask
                else:
                    loss_v = ((value_pred - value_target) ** 2) * agt_mask
                loss_c = loss_v.sum() / agt_mask.sum()

            loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        super(MAPPO_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        # define mindspore trainers
        self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, config.vf_coef, config.ent_coef,
                                                   config.clip_range, config.use_value_clip, config.value_clip_range,
                                                   config.use_huber_loss)
        if self.args.use_grad_norm:
            self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer, clip_type=config.clip_type,
                                                             clip_value=config.max_grad_norm)
        else:
            self.policy_train = nn.TrainOneStepCell(self.loss_net, self.optimizer)
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

    def lr_decay(self, i_step):
        if self.use_linear_lr_decay:
            update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        values = Tensor(sample['values'])
        returns = Tensor(sample['returns'])
        advantages = Tensor(sample['advantages'])
        log_pi_old = Tensor(sample['log_pi_old'])
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        loss = self.policy_train(batch_size, state, obs, actions, log_pi_old, values, returns, advantages, agent_mask, IDs)

        # Logger
        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "learning_rate": lr,
            "loss": loss.asnumpy()
        }

        return info
