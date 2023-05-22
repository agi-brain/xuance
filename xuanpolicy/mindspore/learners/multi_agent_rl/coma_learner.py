"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: MindSpore
"""
from xuanpolicy.mindspore.learners import *


class COMA_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(COMA_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.expand_dims = ops.ExpandDims()

        def construct(self, o, a, agt_mask, ids, q_eval, q_eval_a):
            _, act_prob_ = self._backbone(o, ids)
            act_prob = act_prob_[:, :-1]
            pi_log_prob = self._backbone.actor.log_prob(value=a, probs=act_prob_)[:, :-1]
            baseline = (act_prob * q_eval).sum(-1)

            advantages = q_eval_a - baseline
            loss_coma = -((advantages * pi_log_prob) * agt_mask[:, :-1]).sum() / agt_mask[:, :-1].sum()
            return loss_coma

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(COMA_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.expand_dims = ops.ExpandDims()
            self.mse_loss = nn.MSELoss()

        def construct(self, bs, a_t, agt_mask_t, t, critic_in, tar_q):
            q_eval_t = self._backbone.critic(critic_in)
            q_eval_a_t = GatherD()(q_eval_t, -1, self.expand_dims(a_t, -1).astype(ms.int32)).view(bs, 1, self.n_agents)
            q_eval_a_t *= agt_mask_t
            target_t = tar_q[:, t:t + 1]

            loss_c = self.mse_loss(logits=q_eval_a_t, labels=target_t)
            return loss_c

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Sequence[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.td_lambda = config.td_lambda
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(COMA_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        self.iterations_actor = self.iterations
        self.iterations_critic = 0
        # create loss net and set trainer
        self.zeros_like = ops.ZerosLike()
        self.zeros = ops.Zeros()
        self.actor_loss_net = self.ActorNetWithLossCell(policy, self.n_agents)
        self.actor_train = TrainOneStepCellWithGradClip(self.actor_loss_net, self.optimizer['actor'], clip_type=config.clip_type, clip_value=config.clip_grad)
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy, self.n_agents)
        self.critic_train = TrainOneStepCellWithGradClip(self.critic_loss_net, self.optimizer['critic'], clip_type=config.clip_type, clip_value=config.clip_grad)
        self.critic_train.set_train()

    def build_td_lambda(self, rewards, terminated, agent_mask, target_q_a, max_step_len):
        returns = self.zeros(target_q_a.shape, ms.float32)
        if self.args.consider_terminal_states:
            returns[:, -1] = target_q_a[:, -1] * (1 - terminated.sum(axis=1))
            for t in range(max_step_len - 2, -1, -1):
                returns[:, t] = self.td_lambda * self.gamma * returns[:, t + 1] + (
                        rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_q_a[:, t + 1] * (
                        1 - terminated[:, t])) * agent_mask[:, t]
        else:
            returns[:, -1] = target_q_a[:, -1]
            for t in range(max_step_len - 2, -1, -1):
                returns[:, t] = self.td_lambda * self.gamma * returns[:, t + 1] + (
                        rewards[:, t] + (1 - self.td_lambda) * self.gamma * target_q_a[:, t + 1]) * agent_mask[:, t]
        return returns[:, 0:-1]

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        state_repeat = ops.broadcast_to(self.expand_dims(state, -2), (-1, -1, self.n_agents, -1))
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        actions_onehot = Tensor(sample['actions_onehot'])
        rewards = Tensor(sample['rewards'][:, :-1]).mean(axis=-2)
        terminals = Tensor(sample['terminals'])
        agent_mask = Tensor(sample['agent_mask'])
        batch_size, step_len = obs.shape[0], obs.shape[1]
        IDs = ops.broadcast_to(self.expand_dims(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                                0), (batch_size, step_len, -1, -1))

        # train critic network
        target_critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs)
        target_q_eval = self.policy.target_critic(target_critic_in)
        target_q_a = GatherD()(target_q_eval, -1,
                               self.expand_dims(actions, -1).astype("int32")).view(batch_size, step_len, self.n_agents)
        targets = self.build_td_lambda(rewards, terminals, agent_mask, target_q_a, step_len)

        loss_c_item = 0.0

        q_eval = self.zeros_like(target_q_eval)[:, :-1]
        for t in reversed(range(step_len - 1)):
            agent_mask_t = agent_mask[:, t:t + 1]
            actions_t = self.expand_dims(actions[:, t], -2)
            critic_in = self.policy.build_critic_in(state_repeat, obs, actions_onehot, IDs, t)

            q_eval[:, t:t + 1] = self.policy.critic(critic_in)
            loss_c = self.critic_train(batch_size, actions_t, agent_mask_t, t, critic_in, targets)
            self.iterations_critic += 1
            if self.iterations_critic % self.sync_frequency == 0:
                self.policy.copy_target()
            loss_c_item += loss_c.asnumpy()
        loss_c_item /= (step_len - 1)

        # train actor network
        q_eval_a = GatherD()(q_eval, -1, self.expand_dims(actions[:, :-1], -1).astype(ms.int32))
        q_eval_a = q_eval_a.view(batch_size, step_len-1, self.n_agents)
        loss_coma = self.actor_train(obs, actions, agent_mask, IDs, q_eval, q_eval_a)
        self.iterations_actor += 1

        # Logger
        lr_a = self.scheduler['actor'](self.iterations).asnumpy()
        lr_c = self.scheduler['critic'](self.iterations).asnumpy()
        self.writer.add_scalar("learning_rate_actor", lr_a, self.iterations)
        self.writer.add_scalar("learning_rate_critic", lr_c, self.iterations)
        self.writer.add_scalar("actor_loss", loss_coma.asnumpy(), self.iterations)
        self.writer.add_scalar("critic_loss", loss_c_item, self.iterations)
