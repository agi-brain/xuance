from xuance.mindspore.policies import *
from xuance.mindspore.utils import *
from mindspore.nn.probability.distribution import Normal
import copy


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
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
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

    def construct(self, x: ms.Tensor):
        return self.mu(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.Tensor):
        return self.model(x)[:, 0]


class ActorCriticPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Box)
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class ActorPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Box)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class ActorNet_SAC(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
            layers.extend(mlp)
        self.output = nn.SequentialCell(*layers)
        self.out_mu = nn.Dense(hidden_sizes[0], action_dim)
        self.out_std = nn.Dense(hidden_sizes[0], action_dim)
        self._tanh = ms.ops.Tanh()
        self._exp = ms.ops.Exp()

    def construct(self, x: ms.tensor):
        output = self.output(x)
        mu = self._tanh(self.out_mu(output))
        std = ms.ops.clip_by_value(self.out_std(output), -20, 2)
        std = self._exp(std)
        # dist = Normal(mu, std)
        # return dist
        return mu, std


class CriticNet_SAC(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CriticNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = nn.SequentialCell(*layers)
        self._concat = ms.ops.Concat(-1)

    def construct(self, x: ms.tensor, a: ms.tensor):
        return self.model(self._concat((x, a)))[:, 0]


class SACPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Box)
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        try:
            self.representation_params = self.representation.trainable_params()
        except:
            self.representation_params = []

        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  initialize, activation)
        self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                    initialize, activation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.nor = Normal()

    def action(self, observation: ms.tensor):
        outputs = self.representation(observation)
        # act_dist = self.actor(outputs[0])
        mu, std = self.actor(outputs['state'])
        act_dist = Normal(mu, std)

        return outputs, act_dist

    def Qtarget(self, observation: ms.tensor):
        outputs = self.representation(observation)
        # act_dist = self.target_actor(outputs[0])
        mu, std = self.target_actor(outputs['state'])
        # act_dist = Normal(mu, std)

        # act = act_dist.sample()
        # act_log = act_dist.log_prob(act)
        act = self.nor.sample(mean=mu, sd=std)
        act_log = self.nor.log_prob(act, mu, std)
        return outputs, act_log, self.target_critic(outputs['state'], act)

    def Qaction(self, observation: ms.tensor, action: ms.Tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], action)

    def Qpolicy(self, observation: ms.tensor):
        outputs = self.representation(observation)
        # act_dist = self.actor(outputs['state'])
        mu, std = self.actor(outputs['state'])
        # act_dist = Normal(mu, std)
        
        # act = act_dist.sample()
        # act_log = act_dist.log_prob(act)
        act = self.nor.sample(mean=mu, sd=std)
        act_log = self.nor.log_prob(act, mu, std)
        return outputs, act_log, self.critic(outputs['state'], act)

    def construct(self):
        return super().construct()

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))