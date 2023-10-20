from xuance.mindspore.policies import *
from xuance.mindspore.utils import *
from xuance.mindspore.representations import Basic_Identical
from mindspore.nn.probability.distribution import Normal
import copy


class BasicQhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers_)

    def construct(self, x: ms.tensor):
        return self.model(x)


class BasicQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        q_inputs = self._concat([outputs['state'], agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        q_inputs = self._concat([outputs['state'], agent_ids])
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class Sample(nn.Cell):
    def __init__(self):
        super(Sample, self).__init__()
        self._dist = Normal(dtype=ms.float32)

    def construct(self, mean: ms.tensor, std: ms.tensor):
        return self._dist.sample(mean=mean, sd=std)


class LogProb(nn.Cell):
    def __init__(self):
        super(LogProb, self).__init__()
        self._dist = Normal(dtype=ms.float32)
        self._sum = ms.ops.ReduceSum(keep_dims=False)

    def construct(self, value: ms.tensor, mean: ms.tensor, std: ms.tensor):
        return self._sum(self._dist.log_prob(value, mean, std), -1)


class Entropy(nn.Cell):
    def __init__(self):
        super(Entropy, self).__init__()
        self._dist = Normal(dtype=ms.float32)
        self._sum = ms.ops.ReduceSum(keep_dims=False)

    def construct(self, probs: ms.tensor, std: ms.tensor):
        return self._sum(self._dist.entropy(probs, std), -1)


class ActorNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        # layers.extend(mlp_block(input_shape[0], action_dim, None, nn.ReLU, initialize, device)[0])
        # self.mu = nn.Sequential(*layers)
        # self.logstd = nn.Sequential(*layers)
        self.output = nn.SequentialCell(*layers)
        self.out_mu = nn.Dense(hidden_sizes[0], action_dim)
        self.out_std = nn.Dense(hidden_sizes[0], action_dim)
        self._sigmoid = nn.Sigmoid()
        self._exp = ms.ops.Exp()
        # define the distribution methods
        self.sample = Sample()
        self.log_prob = LogProb()
        self.entropy = Entropy()

    def construct(self, x: ms.tensor):
        output = self.output(x)
        mu = self._sigmoid(self.out_mu(output))
        # std = torch.tanh(self.out_std(output))
        std = self.out_std(output).clip(-20, 1)
        std = self._exp(std)
        return mu, std


class CriticNet(nn.Cell):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: torch.tensor):
        return self.model(x)


class Basic_ISAC_policy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Box)
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation)
        self.target_critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initialize, activation)
        self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self.soft_update(tau=1.0)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu, log_std = self.actor_net(actor_in)
        return outputs, mu, log_std

    def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs[0], actions, agent_ids])
        return self.critic_net(critic_in)

    def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs[0], actions, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs[0], agent_ids])
        mu, std = self.target_actor_net(actor_in)
        return mu, std

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class MASAC_policy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Box)
        super(MASAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
        self.target_critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initialize, activation)
        self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self.soft_update(tau=1.0)
        self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))
        self.broadcast_to_act = ms.ops.BroadcastTo((-1, self.n_agents, -1))

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu, log_std = self.actor_net(actor_in)
        return outputs, mu, log_std

    def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.critic_net(critic_in)

    def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)[0].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs[0], agent_ids])
        mu, std = self.target_actor_net(actor_in)
        return mu, std

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
