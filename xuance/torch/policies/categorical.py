import torch
import numpy as np
import torch.nn as nn
from typing import Sequence, Optional, Callable, Union
from copy import deepcopy
from gym.spaces import Discrete
from xuance.torch import Module, Tensor
from xuance.torch.utils import ModuleType
from xuance.torch.policies.core import CategoricalActorNet as ActorNet
from xuance.torch.policies.core import CategoricalActorNet_SAC as Actor_SAC
from xuance.torch.policies.core import CriticNet
from xuance.torch.policies.core import BasicQhead


def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCriticPolicy(Module):
    """
    Actor-Critic for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the hidden states, action distribution, and values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
            value: The state values output by critic.
        """
        outputs = self.representation(observation)
        a_dist = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        return outputs, a_dist, value[:, 0]


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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the hidden states, action distribution.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
        """
        outputs = self.representation(observation)
        a_dist = self.actor(outputs['state'])
        return outputs, a_dist


class PPGActorCritic(Module):
    """
    Actor-Critic for PPG with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.aux_critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation, device)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the actors representation output, action distribution, values, and auxiliary values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            policy_outputs: The outputs of actor representation.
            a_dist: The distribution of actions output by actor.
            value: The state values output by critic.
            aux_value: The auxiliary values output by aux_critic.
        """
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        aux_critic_outputs = self.aux_critic_representation(observation)
        a_dist = self.actor(policy_outputs['state'])
        value = self.critic(critic_outputs['state'])
        aux_value = self.aux_critic(aux_critic_outputs['state'])
        return policy_outputs, a_dist, value[:, 0], aux_value[:, 0]


class SACDISPolicy(Module):
    """
    Actor-Critic for SAC with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                               normalize, initialize, activation, device)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation, device)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
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
        """
        Returns the output of actor representation and samples of actions.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            outputs: The outputs of the actor representation.
            act_sample: The sampled actions from the distribution output by the actor.
        """
        outputs = self.actor_representation(observation)
        act_dist = self.actor(outputs['state'])
        act_samples = act_dist.stochastic_sample()
        return outputs, act_samples

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """
        Feedforward and calculate the action probabilities, log of action probabilities, and Q-values.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            act_prob: The probabilities of actions.
            log_action_prob: The log of action probabilities.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        act_prob = act_dist.probs
        z = act_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = torch.log(act_prob + z)

        q_1 = self.critic_1(outputs_critic_1['state'])
        q_2 = self.critic_2(outputs_critic_2['state'])
        return act_prob, log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """
        Calculate the action probabilities, log of action probabilities, and Q-values with target networks.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            new_act_prob: The probabilities of actions.
            log_action_prob: The log of action probabilities.
            target_q: The minimum of Q-values calculated by the target critic networks.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        new_act_prob = new_act_dist.probs
        z = new_act_prob == 0.0
        z = z.float() * 1e-8  # avoid log(0)
        log_action_prob = torch.log(new_act_prob + z)

        target_q_1 = self.target_critic_1(outputs_critic_1['state'])
        target_q_2 = self.target_critic_2(outputs_critic_2['state'])
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_prob, log_action_prob, target_q

    def Qaction(self, observation: Union[np.ndarray, dict]):
        """
        Returns the evaluated Q-values for current observations.

        Parameters:
            observation: The original observation.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(outputs_critic_1['state'])
        q_2 = self.critic_2(outputs_critic_2['state'])
        return q_1, q_2

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
