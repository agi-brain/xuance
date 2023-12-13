"""
Multi-Agent TD3

"""
from xuance.mindspore.learners import *


class MATD3_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(MATD3_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._mean = ms.ops.ReduceMean(keep_dims=True)
            self.n_agents = n_agents

        def construct(self, bs, o, ids, agt_mask):
            _, actions_eval = self._backbone(o, ids)
            actions_n_eval = ms.ops.broadcast_to(actions_eval.view(bs, 1, -1), (-1, self.n_agents, -1))
            _, policy_q = self._backbone.Qpolicy(o, actions_n_eval, ids)
            loss_a = -policy_q.mean()
            return loss_a

    class CriticNetWithLossCell_A(nn.Cell):
        def __init__(self, backbone):
            super(MATD3_Learner.CriticNetWithLossCell_A, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()

        def construct(self, o, acts, ids, agt_mask, tar_q):
            _, q_eval = self._backbone.Qaction_A(o, acts, ids)
            td_error = (q_eval - tar_q) * agt_mask
            loss_c = (td_error ** 2).sum() / agt_mask.sum()
            return loss_c

    class CriticNetWithLossCell_B(nn.Cell):
        def __init__(self, backbone):
            super(MATD3_Learner.CriticNetWithLossCell_B, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()

        def construct(self, o, acts, ids, agt_mask, tar_q):
            _, q_eval = self._backbone.Qaction_B(o, acts, ids)
            td_error = (q_eval - tar_q) * agt_mask
            loss_c = (td_error ** 2).sum() / agt_mask.sum()
            return loss_c

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Sequence[nn .exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100,
                 delay: int = 3
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.delay = delay
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(MATD3_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic_A': optimizer[1],
            'critic_B': optimizer[2]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic_A': scheduler[1],
            'critic_B': scheduler[2]
        }
        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
        self.actor_train.set_train()
        self.critic_loss_net_A = self.CriticNetWithLossCell_A(policy)
        self.critic_train_A = nn.TrainOneStepCell(self.critic_loss_net_A, self.optimizer['critic_A'])
        self.critic_train_A.set_train()
        self.critic_loss_net_B = self.CriticNetWithLossCell_B(policy)
        self.critic_train_B = nn.TrainOneStepCell(self.critic_loss_net_B, self.optimizer['critic_B'])
        self.critic_train_B.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        obs_next = Tensor(sample['obs_next'])
        rewards = Tensor(sample['rewards'])
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        # train critic
        actions_next = self.policy.target_actor(obs_next, IDs)
        actions_next_n = ms.ops.broadcast_to(actions_next.view(batch_size, 1, -1), (-1, self.n_agents, -1))
        _, target_q = self.policy.Qtarget(obs_next, actions_next_n, IDs)
        q_target = rewards + (1 - terminals) * self.args.gamma * target_q

        actions_n = ms.ops.broadcast_to(actions.view(batch_size, 1, -1), (-1, self.n_agents, -1))
        loss_c_A = self.critic_train_A(obs, actions_n, IDs, agent_mask, q_target)
        loss_c_B = self.critic_train_B(obs, actions_n, IDs, agent_mask, q_target)

        # actor update
        if self.iterations % self.delay == 0:
            p_loss = self.actor_train(batch_size, obs, IDs, agent_mask)
            self.policy.soft_update(self.tau)

        lr_a = self.scheduler['actor'](self.iterations).asnumpy()
        lr_c_A = self.scheduler['critic_A'](self.iterations).asnumpy()
        lr_c_B = self.scheduler['critic_B'](self.iterations).asnumpy()

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic_A": lr_c_A,
            "learning_rate_critic_B": lr_c_B,
            "loss_critic_A": loss_c_A.asnumpy(),
            "loss_critic_B": loss_c_B.asnumpy()
        }

        if self.iterations % self.delay == 0:
            info["loss_actor"] = p_loss.asnumpy()

        return info
