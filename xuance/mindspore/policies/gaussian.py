import mindspore as ms
import mindspore.nn as nn
import numpy as np
from xuance.common import Sequence, Optional, Callable, Union
from copy import deepcopy
from gym.spaces import Box
from xuance.torch import Module, Tensor
from xuance.torch.utils import ModuleType
from .core import GaussianActorNet as ActorNet
from .core import CriticNet, GaussianActorNet_SAC


class ActorPolicy(Module):
    def __init__(self,
                 action_space: Box,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        assert isinstance(action_space, Box)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
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
                 action_space: Box,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
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

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class SACPolicy(Module):
    def __init__(self,
                 action_space: Box,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 initialize: Optional[Callable[..., Tensor]] = None,
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

        self.actor = GaussianActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                    initialize, activation)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.nor = Normal()

    def action(self, observation: Tensor):
        outputs = self.representation(observation)
        # act_dist = self.actor(outputs[0])
        mu, std = self.actor(outputs['state'])
        act_dist = Normal(mu, std)

        return outputs, act_dist

    def Qtarget(self, observation: Tensor):
        outputs = self.representation(observation)
        # act_dist = self.target_actor(outputs[0])
        mu, std = self.target_actor(outputs['state'])
        # act_dist = Normal(mu, std)

        # act = act_dist.sample()
        # act_log = act_dist.log_prob(act)
        act = self.nor.sample(mean=mu, sd=std)
        act_log = self.nor.log_prob(act, mu, std)
        return outputs, act_log, self.target_critic(outputs['state'], act)

    def Qaction(self, observation: Tensor, action: Tensor):
        outputs = self.representation(observation)
        return outputs, self.critic(outputs['state'], action)

    def Qpolicy(self, observation: Tensor):
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