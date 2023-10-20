"""
MFAC: Mean Field Actor-Critic
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class MFAC_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents, dim_act):
            super(MFAC_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.dim_act = dim_act
            self.expand_dims = ops.ExpandDims()
            self._one_hot = OneHot()

        def construct(self, bs, o_next, agt_mask, ids):
            _, act_prob_next = self._backbone(o_next, ids)
            actions_next = act_prob_next.argmax(-1)
            log_pi_prob = self.expand_dims(self._backbone.actor_net.log_prob(value=actions_next, probs=act_prob_next),
                                           -1)
            actions_next_onehot = self._one_hot(actions_next.astype(ms.int32), self.dim_act, ms.Tensor(1.0, ms.float32),
                                                ms.Tensor(0.0, ms.float32)).astype(ms.float32)
            act_mean_next = actions_next_onehot.mean(axis=-2)
            act_mean_n_next = ops.broadcast_to(self.expand_dims(act_mean_next, 1), (-1, self.n_agents, -1))
            advantages = self._backbone.target_critic_for_train(o_next, act_mean_n_next, ids)
            actions_select = actions_next.view(bs, self.n_agents, 1)
            advantages = ms.ops.stop_gradient(GatherD()(advantages, -1, actions_select.astype(ms.int32)))

            advantages = log_pi_prob * advantages
            loss_a = -(advantages.sum() / agt_mask.sum())
            return loss_a

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(MFAC_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.expand_dims = ops.ExpandDims()
            self.mse_loss = nn.MSELoss()

        def construct(self, bs, o, a, a_mean, agt_mask, ids, tar_q):
            q_eval = self._backbone.critic(o, a_mean, ids)
            q_eval_a = GatherD()(q_eval, -1, a.view(bs, self.n_agents, 1).astype(ms.int32))
            td_error = (q_eval_a - tar_q) * agt_mask
            loss_c = (td_error ** 2).mean()
            return loss_c

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(MFAC_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        self.bmm = ops.BatchMatMul()
        self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents, self.dim_act)
        self.actor_train = TrainOneStepCellWithGradClip(self.actor_loss_net, self.optimizer['actor'],
                                                        clip_type=config.clip_type, clip_value=config.clip_grad)
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy, self.n_agents)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, self.optimizer['critic'])
        self.critic_train.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        obs_next = Tensor(sample['obs_next'])
        act_mean = Tensor(sample['act_mean'])
        # act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
        rewards = Tensor(sample['rewards'])
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        act_mean_n = ops.broadcast_to(self.expand_dims(act_mean, 1), (-1, self.n_agents, -1))

        # train critic network
        target_pi_next = self.policy.target_actor(obs_next, IDs)
        actions_next = target_pi_next.argmax(-1)
        actions_next_onehot = self.onehot_action(actions_next, self.dim_act).astype(ms.float32)
        act_mean_next = actions_next_onehot.mean(axis=-2)
        act_mean_n_next = ops.broadcast_to(self.expand_dims(act_mean_next, 1), (-1, self.n_agents, -1))

        q_eval_next = self.policy.target_critic(obs_next, act_mean_n_next, IDs)
        shape = q_eval_next.shape
        v_mf = self.bmm(q_eval_next.view(-1, 1, shape[-1]), target_pi_next.view(-1, shape[-1], 1))
        v_mf = v_mf.view(tuple(list(shape[0:-1]) + [1]))
        if self.args.consider_terminal_states:
            q_target = rewards + (1 - terminals) * self.args.gamma * v_mf
        else:
            q_target = rewards + self.args.gamma * v_mf
        q_target = ops.stop_gradient(q_target)
        loss_c = self.critic_loss_net(batch_size, obs, actions, act_mean_n, agent_mask, IDs, q_target)

        # train actor network
        loss_a = self.actor_train(batch_size, obs_next, agent_mask, IDs)

        self.policy.soft_update(self.tau)
        # Logger
        lr_a = self.scheduler['actor'](self.iterations)
        lr_c = self.scheduler['critic'](self.iterations)
        self.writer.add_scalar("learning_rate_actor", lr_a.asnumpy(), self.iterations)
        self.writer.add_scalar("learning_rate_critic", lr_c.asnumpy(), self.iterations)
        self.writer.add_scalar("actor_loss", loss_a.asnumpy(), self.iterations)
        self.writer.add_scalar("critic_loss", loss_c.asnumpy(), self.iterations)
