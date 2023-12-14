"""
Weighted QMIX
Paper link:
https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class WQMIX_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agent, agent_name, alpha):
            super(WQMIX_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self.n_agent = n_agent
            self.agent = agent_name
            self._backbone = backbone
            self.alpha = alpha

        def construct(self, s, o, ids, a, label, agt_mask):
            # calculate Q_tot
            _, action_max, q_eval = self._backbone(o, ids)
            action_max = action_max.view(-1, self.n_agent, 1)
            q_eval_a = GatherD()(q_eval, -1, a)
            q_tot_eval = self._backbone.Q_tot(q_eval_a * agt_mask, s)

            # calculate centralized Q
            q_centralized_eval = self._backbone.q_centralized(o, ids)
            q_centralized_eval_a = GatherD()(q_centralized_eval, -1, action_max)
            q_tot_centralized = self._backbone.q_feedforward(q_centralized_eval_a * agt_mask, s)
            td_error = q_tot_eval - label

            # calculate weights
            ones = ops.ones_like(td_error)
            w = ones * self.alpha
            if self.agent == "CWQMIX":
                condition_1 = ((action_max == a).astype(ms.float32) * agt_mask).astype(ms.bool_).all(axis=1)
                condition_2 = label > q_tot_centralized
                conditions = ops.logical_or(condition_1, condition_2)
                w = ms.numpy.where(conditions, ones, w)
            elif self.agent == "OWQMIX":
                condition = td_error < 0
                w = ms.numpy.where(condition, ones, w)
            else:
                AttributeError("You have assigned an unexpected WQMIX learner!")

            loss_central = ((q_tot_centralized - label) ** 2).sum() / agt_mask.sum()
            loss_qmix = (w * (td_error ** 2)).mean()
            loss = loss_qmix + loss_central
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.alpha = config.alpha
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(WQMIX_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        # build train net
        self._mean = ops.ReduceMean(keep_dims=False)
        self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents, self.args.agent, self.alpha)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions']).view(-1, self.n_agents, 1).astype(ms.int32)
        state_next = Tensor(sample['state_next'])
        obs_next = Tensor(sample['obs_next'])
        rewards = self._mean(Tensor(sample['rewards']), 1)
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=True)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))
        # calculate y_i
        if self.args.double_q:
            _, action_next_greedy, _ = self.policy(obs_next, IDs)
            action_next_greedy = self.expand_dims(action_next_greedy, -1).astype(ms.int32)
        else:
            q_next_eval = self.policy.target_Q(obs_next, IDs)
            action_next_greedy = q_next_eval.argmax(axis=-1, keepdims=True)
        q_eval_next_centralized = GatherD()(self.policy.target_q_centralized(obs_next, IDs), -1, action_next_greedy)
        q_tot_next_centralized = self.policy.target_q_feedforward(q_eval_next_centralized*agent_mask, state_next)

        target_value = rewards + (1 - terminals) * self.args.gamma * q_tot_next_centralized

        # calculate losses and train
        loss = self.policy_train(state, obs, IDs, actions, target_value, agent_mask)
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "learning_rate": lr,
            "loss": loss.asnumpy()
        }

        return info
