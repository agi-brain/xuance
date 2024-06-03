import torch
import numpy as np
import torch.nn as nn
from typing import Sequence, Optional, Callable, Union
from copy import deepcopy
from gym.spaces import Space, Discrete
from xuance.torch import Module, Tensor
from xuance.torch.utils import ModuleType
from xuance.torch.policies.core import BasicQhead
from xuance.torch.policies.core import BasicRecurrent
from xuance.torch.policies.core import DuelQhead
from xuance.torch.policies.core import C51Qhead
from xuance.torch.policies.core import QRDQNhead
from xuance.torch.policies.core import ActorNet
from xuance.torch.policies.core import CriticNet


class BasicQnetwork(Module):
    """
    The base class to implement DQN based policy

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
        """
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            targetQ: The evaluated Q-values output by target Q-network.
        """
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = targetQ.argmax(dim=-1)
        return outputs_target, argmax_action.detach(), targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class DuelQnetwork(Module):
    """
    The policy for deep dueling Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                    normalize, initialize, activation, device)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
        """
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            targetQ: The evaluated Q-values output by target Q-network.
        """
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = targetQ.argmax(dim=-1)
        return outputs_target, argmax_action, targetQ

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class NoisyQnetwork(Module):
    """
    The policy for noisy deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = deepcopy(self.eval_Qhead)
        self.noise_scale = 0.0
        self.eval_noise_parameter = []
        self.target_noise_parameter = []

    def update_noise(self, noisy_bound: float = 0.0):
        """Updates the noises for network parameters."""
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.parameters():
            self.eval_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)
            self.target_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
        """
        outputs = self.representation(observation)
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Qhead.parameters(), self.eval_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            targetQ: The evaluated Q-values output by target Q-network.
        """
        outputs = self.target_representation(observation)
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.target_Qhead.parameters(), self.target_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = targetQ.argmax(dim=-1)
        return outputs, argmax_action, targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class C51Qnetwork(Module):
    """
    The policy for C51 distributional deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        atom_num (int): The number of atoms.
        v_min (float): The lower bound of value distribution.
        v_max (float): The upper bound of value distribution.
        representation (Module): The representation module.
        hidden_size: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 v_min: float,
                 v_max: float,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(C51Qnetwork, self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.v_min = v_min
        self.v_max = v_max
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim, self.atom_num,
                                   hidden_size, normalize, initialize, activation, device)
        self.target_Zhead = deepcopy(self.eval_Zhead)
        self.supports = torch.nn.Parameter(torch.linspace(self.v_min, self.v_max, self.atom_num),
                                           requires_grad=False).to(device)
        self.deltaz = (v_max - v_min) / (atom_num - 1)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Z-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            eval_Z: The evaluated Z-values.
        """
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = (self.supports * eval_Z).sum(-1)
        argmax_action = eval_Q.argmax(dim=-1)
        return outputs, argmax_action, eval_Z

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Z-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            target_Z: The evaluated Z-values output by target Z-network.
        """
        outputs_target = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs_target['state'])
        target_Q = (self.supports * target_Z).sum(-1)
        argmax_action = target_Q.argmax(dim=-1)
        return outputs_target, argmax_action, target_Z

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
            tp.data.copy_(ep)


class QRDQN_Network(Module):
    """
    The policy for quantile regression deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        quantile_num (int): The number of quantiles.
        representation (Module): The representation module.
        hidden_size: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim, self.quantile_num,
                                    hidden_size, normalize, initialize, activation, device)
        self.target_Zhead = deepcopy(self.eval_Zhead)

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Z-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            eval_Z: The evaluated Z-values.
        """
        outputs = self.representation(observation)
        eval_Z = self.eval_Zhead(outputs['state'])
        eval_Q = eval_Z.mean(dim=-1)
        argmax_action = eval_Q.argmax(dim=-1)
        return outputs, argmax_action, eval_Z

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Z-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            target_Z: The evaluated Z-values output by target Z-network.
        """
        outputs = self.target_representation(observation)
        target_Z = self.target_Zhead(outputs['state'])
        target_Q = target_Z.mean(dim=-1)
        argmax_action = target_Q.argmax(dim=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Zhead.parameters(), self.target_Zhead.parameters()):
            tp.data.copy_(ep)


class DDPGPolicy(Module):
    """
    The policy of deep deterministic policy gradient.

    Args:
        action_space (Space): The action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): List of hidden units for actor network.
        critic_hidden_size (Sequence[int]): List of hidden units for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
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
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes
        # create networks
        self.actor_representation = representation
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic_representation = deepcopy(representation)
        self.critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                normalize, initialize, activation, device)
        # create target networks
        self.target_actor_representation = deepcopy(self.actor_representation)
        self.target_actor = deepcopy(self.actor)
        self.target_critic_representation = deepcopy(self.critic_representation)
        self.target_critic = deepcopy(self.critic)

        # parameters
        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the actor representations, and the actions.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The output of the actor representations.
            act: The actions calculated by the actor.
        """
        outputs = self.actor_representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values via target networks."""
        outputs_actor = self.target_actor_representation(observation)
        outputs_critic = self.target_critic_representation(observation)
        act = self.target_actor(outputs_actor['state'])
        q_ = self.target_critic(torch.concat([outputs_critic['state'], act], dim=-1))
        return q_[:, 0]

    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        """Returns the evaluated Q-values of state-action pairs."""
        outputs = self.critic_representation(observation)
        q = self.critic(torch.concat([outputs['state'], action], dim=-1))
        return q[:, 0]

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values by calculating actions via actor networks."""
        outputs_actor = self.actor_representation(observation)
        act = self.actor(outputs_actor['state'])
        outputs_critic = self.critic_representation(observation)
        q_eval = self.critic(torch.concat([outputs_critic['state'], act], dim=-1))
        return q_eval[:, 0]

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class TD3Policy(Module):
    """
    The policy of twin delayed deep deterministic policy gradient.

    Args:
        action_space (Space): The action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): List of hidden units for actor network.
        critic_hidden_size (Sequence[int]): List of hidden units for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
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
        super(TD3Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.critic_A_representation = deepcopy(representation)
        self.critic_B_representation = deepcopy(representation)

        self.target_actor_representation = deepcopy(representation)
        self.target_critic_A_representation = deepcopy(representation)
        self.target_critic_B_representation = deepcopy(representation)

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic_A = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_B = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic_A = deepcopy(self.critic_A)
        self.target_critic_B = deepcopy(self.critic_B)

        # parameters
        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_A_representation.parameters()) + list(
            self.critic_A.parameters()) + list(self.critic_B_representation.parameters()) + list(
            self.critic_B.parameters())

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the actor representations, and the actions.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The output of the actor representations.
            act: The actions calculated by the actor.
        """
        outputs = self.actor_representation(observation)
        act = self.actor(outputs['state'])
        return outputs, act

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values via target networks."""
        outputs_actor = self.target_actor_representation(observation)
        outputs_critic_A = self.target_critic_A_representation(observation)
        outputs_critic_B = self.target_critic_B_representation(observation)
        act = self.target_actor(outputs_actor['state'])
        noise = (torch.randn_like(act) * 0.2).clamp(-0.5, 0.5)
        act = (act + noise).clamp(-1, 1)

        qa = self.target_critic_A(torch.concat([outputs_critic_A['state'], act], dim=-1))
        qb = self.target_critic_B(torch.concat([outputs_critic_B['state'], act], dim=-1))
        min_q = torch.min(qa, qb)
        return min_q[:, 0]

    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        """Returns the evaluated Q-values of state-action pairs."""
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        q_eval_a = self.critic_A(torch.concat([outputs_critic_A['state'], action], dim=-1))
        q_eval_b = self.critic_B(torch.concat([outputs_critic_B['state'], action], dim=-1))
        return q_eval_a[:, 0], q_eval_b[:, 0]

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values by calculating actions via actor networks."""
        outputs_actor = self.actor_representation(observation)
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        act = self.actor(outputs_actor['state'])
        q_eval_a = self.critic_A(torch.concat([outputs_critic_A['state'], act], dim=-1)).unsqueeze(dim=1)
        q_eval_b = self.critic_B(torch.concat([outputs_critic_B['state'], act], dim=-1)).unsqueeze(dim=1)
        return (q_eval_a + q_eval_b) / 2.0

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A_representation.parameters(), self.target_critic_A_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A.parameters(), self.target_critic_A.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B_representation.parameters(), self.target_critic_B_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B.parameters(), self.target_critic_B.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class PDQNPolicy(Module):
    """
    The policy of parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize, initialize, activation, device)
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 normalize, initialize, activation, activation_action, device)
        self.target_conactor = deepcopy(self.conactor)
        self.target_qnetwork = deepcopy(self.qnetwork)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        target_q = self.target_qnetwork(input_q)
        return target_q

    def Qeval(self, state, action):
        input_q = torch.cat((state, action), dim=1)
        eval_q = self.qnetwork(input_q)
        return eval_q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = torch.cat((state, conact), dim=1)
        policy_q = torch.sum(self.qnetwork(input_q))
        return policy_q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.conactor.parameters(), self.target_conactor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MPDQNPolicy(PDQNPolicy):
    """
    The policy of multi-pass parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(MPDQNPolicy, self).__init__(observation_space, action_space, representation,
                                          conactor_hidden_size, qnetwork_hidden_size,
                                          normalize, initialize, activation, activation_action, device)
        self.obs_size = self.observation_space.shape[0]
        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qeval(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(action)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        batch_size = state.shape[0]
        Q = []
        input_q = torch.cat((state, torch.zeros_like(conact)), dim=1)
        input_q = input_q.repeat(self.num_disact, 1)
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = conact[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = eval_q.unsqueeze(1)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q


class SPDQNPolicy(PDQNPolicy):
    """
    The policy of split parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(SPDQNPolicy, self).__init__(observation_space, action_space, representation,
                                          conactor_hidden_size, qnetwork_hidden_size,
                                          normalize, initialize, activation, activation_action, device)
        self.qnetwork = nn.ModuleList()
        for k in range(self.num_disact):
            self.qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size,
                           normalize, initialize, activation, device))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 normalize, initialize, activation, activation_action, device)
        self.target_conactor = deepcopy(self.conactor)
        self.target_qnetwork = deepcopy(self.qnetwork)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)

    def Qtarget(self, state, action):
        target_Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.target_qnetwork[i](input_q)
            target_Q.append(eval_q)
        target_Q = torch.cat(target_Q, dim=1)
        return target_Q

    def Qeval(self, state, action):
        Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q

    def Qpolicy(self, state):
        conacts = self.conactor(state)
        Q = []
        for i in range(self.num_disact):
            conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = torch.cat((state, conact), dim=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = torch.cat(Q, dim=1)
        return Q


class DRQNPolicy(Module):
    """
    The policy of deep recurrent Q-networks.

    Args:
        action_space: The action space.
        representation: The representation module.
        **kwargs: The other arguments.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 **kwargs):
        super(DRQNPolicy, self).__init__()
        self.device = kwargs['device']
        self.recurrent_layer_N = kwargs['recurrent_layer_N']
        self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        kwargs["input_dim"] = self.representation.output_shapes['state'][0]
        kwargs["action_dim"] = self.action_dim
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.cnn = True if self.representation._get_name() == "Basic_CNN" else False
        self.eval_Qhead = BasicRecurrent(**kwargs)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    def forward(self, observation: Union[np.ndarray, dict], *rnn_hidden: Tensor):
        """
        Returns the output of the representation, greedy actions, the evaluated Q-values and the RNN hidden states.

        Parameters:
            observation: The original observation input.
            rnn_hidden: The RNN hidden state.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
            (hidden_states, cell_states): The updated RNN hidden states.
        """
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.representation(observation.reshape((-1,) + obs_shape[-3:]))
            outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            outputs = self.representation(observation)
        if self.lstm:
            hidden_states, cell_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
        else:
            hidden_states, evalQ = self.eval_Qhead(outputs['state'], rnn_hidden[0])
            cell_states = None
        argmax_action = evalQ[:, -1].argmax(dim=-1)
        return outputs, argmax_action, evalQ, (hidden_states, cell_states)

    def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: Tensor):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.target_representation(observation.reshape((-1,) + obs_shape[-3:]))
            outputs['state'] = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            outputs = self.target_representation(observation)
        if self.lstm:
            hidden_states, cell_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0], rnn_hidden[1])
        else:
            hidden_states, targetQ = self.target_Qhead(outputs['state'], rnn_hidden[0])
            cell_states = None
        argmax_action = targetQ.argmax(dim=-1)
        return outputs, argmax_action, targetQ.detach(), (hidden_states, cell_states)

    def init_hidden(self, batch):
        hidden_states = torch.zeros(size=(self.recurrent_layer_N, batch, self.rnn_hidden_dim)).to(self.device)
        cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
        return hidden_states, cell_states

    def init_hidden_item(self, rnn_hidden, i):
        if self.lstm:
            rnn_hidden[0][:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
            rnn_hidden[1][:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
            return rnn_hidden
        else:
            rnn_hidden[:, i] = torch.zeros(size=(self.recurrent_layer_N, self.rnn_hidden_dim)).to(self.device)
            return rnn_hidden

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
