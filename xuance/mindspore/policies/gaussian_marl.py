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


class ActorNet(nn.Cell):
    class Sample(nn.Cell):
        def __init__(self, log_std):
            super(ActorNet.Sample, self).__init__()
            self._dist = Normal(dtype=ms.float32)
            self.logstd = log_std
            self._exp = ms.ops.Exp()

        def construct(self, mean: ms.tensor):
            return self._dist.sample(mean=mean, sd=self._exp(self.logstd))

    class LogProb(nn.Cell):
        def __init__(self, log_std):
            super(ActorNet.LogProb, self).__init__()
            self._dist = Normal(dtype=ms.float32)
            self.logstd = log_std
            self._exp = ms.ops.Exp()
            self._sum = ms.ops.ReduceSum(keep_dims=False)

        def construct(self, value: ms.tensor, probs: ms.tensor):
            return self._sum(self._dist.log_prob(value, probs, self._exp(self.logstd)), -1)

    class Entropy(nn.Cell):
        def __init__(self, log_std):
            super(ActorNet.Entropy, self).__init__()
            self._dist = Normal(dtype=ms.float32)
            self.logstd = log_std
            self._exp = ms.ops.Exp()
            self._sum = ms.ops.ReduceSum(keep_dims=False)

        def construct(self, probs: ms.tensor):
            return self._sum(self._dist.entropy(probs, self._exp(self.logstd)), -1)

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
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
        self.mu = nn.SequentialCell(*layers)
        self._ones = ms.ops.Ones()
        self.logstd = ms.Parameter(-self._ones((action_dim,), ms.float32))
        # define the distribution methods
        self.sample = self.Sample(self.logstd)
        self.log_prob = self.LogProb(self.logstd)
        self.entropy = self.Entropy(self.logstd)

    def construct(self, x: ms.tensor):
        return self.mu(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents, )
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class MAAC_Policy(nn.Cell):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Cell,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation[0]
        self.representation_critic = representation[1]
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.actor = ActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                              actor_hidden_size, normalize, initialize, activation)
        dim_input_critic = self.representation_critic.output_shapes['state'][0]
        self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor,
                  *rnn_hidden: ms.tensor, **kwargs):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor(actor_input)
        return rnn_hidden, mu_a

    def get_values(self, critic_in: ms.tensor, agent_ids: ms.tensor, *rnn_hidden: ms.tensor, **kwargs):
        shape_obs = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
            outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
            outputs['state'] = outputs['state'].view(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = self._concat([outputs['state'], agent_ids])
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: ms.tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


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
        super(Basic_ISAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
        self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation)
        self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                           normalize, initialize, activation)
        self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self.soft_update(tau=1.0)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor_net(actor_in)
        return outputs, mu_a

    def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.target_actor_net(actor_in)
        return mu_a

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
        super(MASAC_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents
        self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation)
        self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                           normalize, initialize, activation)
        self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self.soft_update(tau=1.0)
        self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))
        self.broadcast_to_act = ms.ops.BroadcastTo((-1, self.n_agents, -1))

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor_net(actor_in)
        return outputs, mu_a

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
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.target_actor_net(actor_in)
        return mu_a

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
