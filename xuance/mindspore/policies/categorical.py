import mindspore as ms
import mindspore.nn as nn
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import Module, Tensor
from xuance.mindspore.utils import ModuleType
from .core import CategoricalActorNet as ActorNet
# from .core import CategoricalActorNet_SAC as Actor_SAC
from .core import CriticNet, BasicQhead


def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorPolicy(Module):
    """
    Actor for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a
    
    
class ActorCriticPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
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

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class PPGActorCritic(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.aux_critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation)

    def construct(self, observation: Tensor):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a, v, aux_v


# class SACDISPolicy(Module):
#     def __init__(self,
#                  action_space: Space,
#                  representation: ModuleType,
#                  actor_hidden_size: Sequence[int],
#                  critic_hidden_size: Sequence[int],
#                  normalize: Optional[ModuleType] = None,
#                  initialize: Optional[Callable[..., Tensor]] = None,
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

#     def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
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



class SACDISPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        # assert isinstance(action_space, Box)
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_critic = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        try:
            self.representation_params = self.representation.trainable_params()
        except:
            self.representation_params = []

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                       initialize, activation)
        self.target_representation_critic = deepcopy(self.representation_critic)
        self.target_critic = deepcopy(self.critic)
        self.actor_params = self.representation_params + self.actor.trainable_params()
        self._log = ms.ops.Log()

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        act_prob = self.actor(outputs["state"])
        return outputs, act_prob

    def action(self, observation: Tensor):
        outputs = self.representation(observation)
        act_prob = self.actor(outputs[0])
        return outputs, act_prob

    def Qtarget(self, observation: Tensor):
        outputs = self.representation(observation)
        outputs_critic = self.target_representation_critic(observation)
        act_prob = self.actor(outputs['state'])
        log_action_prob = self._log(act_prob + 1e-10)
        return act_prob, log_action_prob, self.target_critic(outputs_critic['state'])

    def Qaction(self, observation: Tensor):
        outputs = self.representation_critic(observation)
        return outputs, self.critic(outputs['state'])

    def Qpolicy(self, observation: Tensor):
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
