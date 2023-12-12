"""
Independent Q-learning (IQL)
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class IQL_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(IQL_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone

        def construct(self, o, ids, a, label, agt_mask):
            _, _, q_eval = self._backbone(o, ids)
            q_eval_a = GatherD()(q_eval, -1, a)
            td_error = (q_eval_a - label) * agt_mask
            loss = (td_error ** 2).sum() / agt_mask.sum()
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
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(IQL_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        # build train net
        self.loss_net = self.PolicyNetWithLossCell(policy)
        self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
        self.policy_train.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions']).view(-1, self.n_agents, 1).astype(ms.int32)
        obs_next = Tensor(sample['obs_next'])
        rewards = Tensor(sample['rewards'])
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))
        # calculate the target values
        _, q_next = self.policy.target_Q(obs_next, IDs)
        if self.args.double_q:
            _, action_next_greedy, q_next_eval = self.policy(obs_next, IDs)
            action_next_greedy = self.expand_dims(action_next_greedy, -1).astype(ms.int32)
            q_next_a = GatherD()(q_next, -1, action_next_greedy)
        else:
            q_next_a = q_next.max(axis=-1, keepdims=True).values

        q_target = rewards + (1-terminals) * self.args.gamma * q_next_a
        # train the model
        loss = self.policy_train(obs, IDs, actions, q_target, agent_mask)

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        # log
        lr = self.scheduler(self.iterations).asnumpy()

        info = {
            "learning_rate": lr,
            "loss_Q": loss.asnumpy()
        }

        return info
