from xuance.mindspore.policies import *
from xuance.mindspore.utils import *
from mindspore.nn.probability.distribution import Categorical
import copy


class ActorNet(nn.Cell):
    class Sample(nn.Cell):
        def __init__(self):
            super(ActorNet.Sample, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs: ms.tensor):
            return self._dist.sample(probs=probs).astype("int32")

    class LogProb(nn.Cell):
        def __init__(self):
            super(ActorNet.LogProb, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, value, probs):
            return self._dist._log_prob(value=value, probs=probs)

    class Entropy(nn.Cell):
        def __init__(self):
            super(ActorNet.Entropy, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs):
            return self._dist.entropy(probs=probs)

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Softmax, None)[0])
        self.model = nn.SequentialCell(*layers)
        self.sample = self.Sample()
        self.log_prob = self.LogProb()
        self.entropy = self.Entropy()

    def construct(self, x: ms.Tensor):
        return self.model(x)


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
        assert isinstance(action_space, Discrete)
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
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
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class PPGActorCritic(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.actor_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.aux_critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a, v, aux_v


# class SACDISPolicy(nn.Cell):
#     def __init__(self,
#                  action_space: Space,
#                  representation: ModuleType,
#                  actor_hidden_size: Sequence[int],
#                  critic_hidden_size: Sequence[int],
#                  normalize: Optional[ModuleType] = None,
#                  initialize: Optional[Callable[..., ms.Tensor]] = None,
#                  activation: Optional[ModuleType] = None):
#         assert isinstance(action_space, Discrete)
#         super(SACDISPolicy, self).__init__()
#         self.action_dim = action_space.n
#         self.representation = representation
#         self.representation_info_shape = self.representation.output_shapes
#         try:
#             self.representation_params = self.representation.trainable_params()
#         except:
#             self.representation_params = []

#         self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
#                               normalize, initialize, activation)
#         self.critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
#                                     initialize, activation)
#         self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
#                                      normalize, initialize, activation)
#         self.target_critic = CriticNet_SAC(representation.output_shapes['state'][0], self.action_dim,
#                                            critic_hidden_size, initialize, activation)
#         self.actor_params = self.representation_params + self.actor.trainable_params()
#         self._unsqueeze = ms.ops.ExpandDims()
#         self._Categorical = Categorical(dtype=ms.float32)
#         self.soft_update(tau=1.0)

#     def action(self, observation: Union[np.ndarray, dict]):
#         outputs = self.representation(observation)
#         act_dist = self.actor(outputs[0])
#         return outputs, act_dist

#     def Qtarget(self, observation: Union[np.ndarray, dict]):
#         outputs = self.representation(observation)
#         act_dist = self.target_actor(outputs['state'])
#         act = self._Categorical.sample(probs=act_dist)
#         act_log = self._Categorical.log_prob(value=act, probs=act_dist)
#         act = self._unsqueeze(act, -1)
#         return act_log, self.target_critic(outputs['state'], act)

#     def Qaction(self, observation: Union[np.ndarray, dict], action: ms.Tensor):
#         outputs = self.representation(observation)
#         action = self._unsqueeze(action, -1)
#         return outputs, self.critic(outputs['state'], action)

#     def Qpolicy(self, observation: Union[np.ndarray, dict]):
#         outputs = self.representation(observation)
#         act_dist = self.actor(outputs['state'])
#         act = self._Categorical.sample(probs=act_dist)
#         act_log = self._Categorical.log_prob(value=act, probs=act_dist)
#         act = self._unsqueeze(act, -1)
#         return act_log, self.critic(outputs['state'], act)

#     def construct(self):
#         return super().construct()

#     def soft_update(self, tau=0.005):
#         for ep, tp in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
#             tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
#         for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
#             tp.assign_value((tau * ep.data + (1 - tau) * tp.data))

class CriticNet_SACDIS(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CriticNet_SACDIS, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, None, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class SACDISPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        # assert isinstance(action_space, Box)
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_critic = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        try:
            self.representation_params = self.representation.trainable_params()
        except:
            self.representation_params = []

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet_SACDIS(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                       initialize, activation)
        self.target_representation_critic = copy.deepcopy(self.representation_critic)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_params = self.representation_params + self.actor.trainable_params()
        self._log = ms.ops.Log()

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        act_prob = self.actor(outputs["state"])
        return outputs, act_prob

    def action(self, observation: ms.tensor):
        outputs = self.representation(observation)
        act_prob = self.actor(outputs[0])
        return outputs, act_prob

    def Qtarget(self, observation: ms.tensor):
        outputs = self.representation(observation)
        outputs_critic = self.target_representation_critic(observation)
        act_prob = self.actor(outputs['state'])
        log_action_prob = self._log(act_prob + 1e-10)
        return act_prob, log_action_prob, self.target_critic(outputs_critic['state'])

    def Qaction(self, observation: ms.tensor):
        outputs = self.representation_critic(observation)
        return outputs, self.critic(outputs['state'])

    def Qpolicy(self, observation: ms.tensor):
        outputs = self.representation(observation)
        outputs_critic = self.representation_critic(observation)
        act_prob = self.actor(outputs['state'])
        log_action_prob = self._log(act_prob + 1e-10)
        return act_prob, log_action_prob, self.critic(outputs_critic['state'])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation_critic.trainable_params(), self.target_representation_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
