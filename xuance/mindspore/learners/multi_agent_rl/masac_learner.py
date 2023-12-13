"""
Multi-agent Soft Actor-critic (MASAC)
Implementation: Pytorch
Creator: Kun Jiang (kjiang@seu.edu.cn)
"""
from xuance.mindspore.learners import *


class MASAC_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents, alpha):
            super(MASAC_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.alpha = alpha

        def construct(self, bs, o, ids, agt_mask):
            _, actions_dist_mu = self._backbone(o, ids)
            actions_eval = self._backbone.actor_net.sample(actions_dist_mu)
            log_pi_a = self._backbone.actor_net.log_prob(actions_eval, actions_dist_mu)
            log_pi_a = ms.ops.expand_dims(log_pi_a, axis=-1)
            loss_a = -(self._backbone.critic_for_train(o, actions_eval, ids) - self.alpha * log_pi_a * agt_mask).sum() / agt_mask.sum()
            return loss_a

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(MASAC_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone

        def construct(self, o, acts, ids, agt_mask, tar_q):
            q_eval = self._backbone.critic_for_train(o, acts, ids)
            td_error = (q_eval - tar_q) * agt_mask
            loss_c = (td_error ** 2).sum() / agt_mask.sum()
            return loss_c

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Sequence[nn.exponential_decay_lr] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(MASAC_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents, self.alpha)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, self.optimizer['critic'])
        self.critic_train.set_train()

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

        actions_next_dist_mu = self.policy.target_actor(obs_next, IDs)
        actions_next = self.policy.target_actor_net.sample(actions_next_dist_mu)
        log_pi_a_next = self.policy.target_actor_net.log_prob(actions_next, actions_next_dist_mu)
        q_next = self.policy.target_critic(obs_next, actions_next, IDs)
        log_pi_a_next = ms.ops.expand_dims(log_pi_a_next, axis=-1)
        q_target = rewards + (1-terminals) * self.args.gamma * (q_next - self.alpha * log_pi_a_next)

        # calculate the loss function
        loss_a = self.actor_train(batch_size, obs, IDs, agent_mask)
        loss_c = self.critic_train(obs, actions, IDs, agent_mask, q_target)

        self.policy.soft_update(self.tau)

        lr_a = self.scheduler['actor'](self.iterations).asnumpy()
        lr_c = self.scheduler['critic'](self.iterations).asnumpy()

        info = {
            "learning_rate_actor": lr_a,
            "loss_actor": loss_a.asnumpy(),
            "learning_rate_critic": lr_c,
            "loss_critic": loss_c.asnumpy()
        }

        return info
