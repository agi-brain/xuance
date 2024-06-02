import torch
import numpy as np
from typing import Sequence, Optional, Callable, Union
from copy import deepcopy
from gym.spaces import Space
from xuance.torch import Module, Tensor
from xuance.torch.utils import ModuleType
from xuance.torch.policies.core import GaussianActorNet as ActorNet
from xuance.torch.policies.core import CriticNet, ActorNet_SAC


class ActorCriticPolicy(Module):
    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v[:, 0]


class ActorPolicy(Module):
    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 fixed_std: bool = True):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        policy_dist = self.actor(outputs['state'])
        return outputs, policy_dist


class PPGActorCritic(Module):
    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a, v, aux_v


class SACPolicy(Module):
    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.activation_action = activation_action()
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = ActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                  normalize, initialize, activation, activation_action, device)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        act_dist = self.actor(outputs_actor['state'])
        act_sample = act_dist.activated_rsample()
        return outputs_actor, act_sample

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        act_sample, act_log = act_dist.activated_rsample_and_logprob()

        q_1 = self.critic_1(torch.concat([outputs_critic_1['state'], act_sample], dim=-1))
        q_2 = self.critic_2(torch.concat([outputs_critic_2['state'], act_sample], dim=-1))
        return act_log, q_1[:, 0], q_2[:, 0]

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        new_act_sample, new_act_log = new_act_dist.activated_rsample_and_logprob()

        target_q_1 = self.target_critic_1(torch.concat([outputs_critic_1['state'], new_act_sample], dim=-1))
        target_q_2 = self.target_critic_2(torch.concat([outputs_critic_2['state'], new_act_sample], dim=-1))
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_log, target_q[:, 0]

    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(torch.concat([outputs_critic_1['state'], action], dim=-1))
        q_2 = self.critic_2(torch.concat([outputs_critic_2['state'], action], dim=-1))
        return q_1[:, 0], q_2[:, 0]

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
