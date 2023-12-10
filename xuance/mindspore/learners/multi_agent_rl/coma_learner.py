"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: MindSpore
"""
from xuance.mindspore.learners import *


class COMA_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(COMA_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.expand_dims = ops.ExpandDims()

        def construct(self, actor_in, ids, epsilon, actions, agent_mask, advantages):
            _, pi_probs = self._backbone(actor_in, ids, epsilon=epsilon)
            pi_a = pi_probs.gather(actions.unsqueeze(-1).astype(ms.int32), -1, -1).squeeze(-1)
            log_pi_a = ops.log(pi_a)
            log_pi_a *= agent_mask
            loss_coma = -(advantages * log_pi_a).sum() / agent_mask.sum()
            return loss_coma

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone, n_agents):
            super(COMA_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self.n_agents = n_agents
            self.expand_dims = ops.ExpandDims()
            self.mse_loss = nn.MSELoss()

        def construct(self, critic_in, actions, agent_mask, target_q):
            _, q_eval = self._backbone.get_values(critic_in)
            q_eval_a = q_eval.gather(actions.unsqueeze(-1).astype(ms.int32), -1, -1).squeeze(-1)
            q_eval_a *= agent_mask
            targets = target_q * agent_mask
            loss_c = ((q_eval_a - targets) ** 2).sum() / agent_mask.sum()
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
        self.td_lambda = config.td_lambda
        self.sync_frequency = sync_frequency
        self.use_global_state = config.use_global_state
        self.mse_loss = nn.MSELoss()
        self._concat = ms.ops.Concat(axis=-1)
        super(COMA_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
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

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        actions_onehot = Tensor(sample['actions_onehot'])
        targets = Tensor(sample['returns']).squeeze(-1)
        agent_mask = Tensor(sample['agent_mask'])
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0), (batch_size, -1, -1))

        # build critic input
        actions_in = ops.broadcast_to(actions_onehot.unsqueeze(1).reshape(batch_size, 1, -1), (-1, self.n_agents, -1))
        actions_in_mask = 1 - self.eye(self.n_agents, self.n_agents, ms.float32)
        actions_in_mask = ops.broadcast_to(actions_in_mask.reshape(-1, 1), (-1, self.dim_act)).reshape(self.n_agents, -1)
        actions_in = actions_in * actions_in_mask.unsqueeze(0)
        if self.use_global_state:
            state = ops.broadcast_to(state.unsqueeze(1), (-1, self.n_agents, -1))
            critic_in = self._concat([state, obs, actions_in])
        else:
            critic_in = self._concat([obs, actions_in])
        # train critic
        loss_c = self.critic_train(critic_in, actions, agent_mask, targets)

        # calculate baselines
        _, pi_probs = self.policy(obs, IDs, epsilon=epsilon)
        _, q_eval = self.policy.get_values(critic_in)
        q_eval_a = q_eval.gather(actions.unsqueeze(-1).astype(ms.int32), -1, -1).squeeze(-1)
        q_eval_a *= agent_mask
        baseline = (pi_probs * q_eval).sum(-1)
        advantages = q_eval_a - baseline
        # train actors
        loss_coma = self.actor_train(obs, IDs, epsilon, actions, agent_mask, advantages)

        # Logger
        lr_a = self.scheduler['actor'](self.iterations).asnumpy()
        lr_c = self.scheduler['critic'](self.iterations).asnumpy()

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "actor_loss": loss_coma.asnumpy(),
            "critic_loss": loss_c.asnumpy(),
        }

        return info
