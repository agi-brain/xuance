"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class MFAC_Learner(LearnerMAS):
    class NetWithLossCell(nn.Cell):
        def __init__(self, backbone, vf_coef, ent_coef):
            super(MFAC_Learner.NetWithLossCell, self).__init__()
            self._backbone = backbone
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef

        def construct(self, obs, actions, returns, advantages, act_mean_n, agt_mask, ids):
            # actor loss
            _, act_probs = self._backbone(obs, ids)
            log_pi = self._backbone.actor.log_prob(value=actions, probs=act_probs).unsqueeze(-1)
            entropy = self._backbone.actor.entropy(act_probs).unsqueeze(-1)

            targets = returns
            value_pred = self._backbone.get_values(obs, act_mean_n, ids)
            td_error = value_pred - targets

            pg_loss = -((advantages * log_pi) * agt_mask).sum() / agt_mask.sum()
            vf_loss = ((td_error ** 2) * agt_mask).sum() / agt_mask.sum()
            entropy_loss = (entropy * agt_mask).sum() / agt_mask.sum()
            loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(MFAC_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bmm = ops.BatchMatMul()
        self.loss_net = self.NetWithLossCell(policy, self.vf_coef, self.ent_coef)
        self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, self.optimizer,
                                                         clip_type=config.clip_type, clip_value=config.max_grad_norm)
        self.policy_train.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        act_mean = Tensor(sample['act_mean'])
        returns = Tensor(sample['returns'])
        agent_mask = Tensor(sample['agent_mask']).astype(ms.float32).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        act_mean_n = ops.broadcast_to(self.expand_dims(act_mean, 1), (-1, self.n_agents, -1))

        targets = returns
        value_pred = self.policy.get_values(obs, act_mean_n, IDs)
        advantages = targets - value_pred
        loss = self.policy_train(obs, actions, returns, advantages, act_mean_n, agent_mask, IDs)

        lr = self.scheduler(self.iterations)

        info = {
            "learning_rate": lr.asnumpy(),
            "loss": loss.asnumpy()
        }

        return info
