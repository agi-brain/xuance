from copy import deepcopy
from gym.spaces import Box
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import Module, Tensor, ops
from xuance.mindspore.utils import ModuleType
from .core import GaussianActorNet as ActorNet
from .core import CriticNet, GaussianActorNet_SAC


class ActorPolicy(Module):
    """
    Actor for stochastic policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 fixed_std: bool = True):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action)

    def construct(self, observation: Union[Tensor, dict]):
        outputs = self.representation(observation)
        a_dist = self.actor(outputs['state'])
        return outputs, a_dist, None


class ActorCriticPolicy(Module):
    """
    Actor-Critic for stochastic policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    def construct(self, observation: Tensor):
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
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])[:, 0]
        return outputs, a, v


class PPGActorCritic(Module):
    """
    Actor-Critic for PPG with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.shape[0]
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation)

    def construct(self, observation: Union[Tensor, dict]):
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
        a_dist = self.actor(policy_outputs['state'])
        value = self.critic(critic_outputs['state'])[:, 0]
        aux_value = self.aux_critic(policy_outputs['state'])[:, 0]
        return policy_outputs, a_dist, value, aux_value


class SACPolicy(Module):
    """
    Actor-Critic for SAC with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(SACPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = GaussianActorNet_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                          normalize, initialize, activation, activation_action)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.actor_parameters = self.actor_representation.trainable_params() + self.actor.trainable_params()
        self.critic_parameters = self.critic_1_representation.trainable_params() + self.critic_1.trainable_params() + \
                                 self.critic_2_representation.trainable_params() + self.critic_2.trainable_params()

    def construct(self, observation: Union[Tensor, dict]):
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
        act_sample = act_dist.activated_rsample()
        return outputs, act_sample

    def Qpolicy(self, observation: Union[Tensor, dict]):
        """
        Feedforward and calculate the log of action probabilities, and Q-values.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            log_action_prob: The log of action probabilities.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        act_sample, log_action_prob = act_dist.activated_rsample_and_logprob()

        q_1 = self.critic_1(ops.cat([outputs_critic_1['state'], act_sample], axis=-1))
        q_2 = self.critic_2(ops.cat([outputs_critic_2['state'], act_sample], axis=-1))
        return log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[Tensor, dict]):
        """
        Calculate the log of action probabilities and Q-values with target networks.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            log_action_prob: The log of action probabilities.
            target_q: The minimum of Q-values calculated by the target critic networks.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        new_act_sample, log_action_prob = new_act_dist.activated_rsample_and_logprob()

        target_q_1 = self.target_critic_1(ops.cat([outputs_critic_1['state'], new_act_sample], axis=-1))
        target_q_2 = self.target_critic_2(ops.cat([outputs_critic_2['state'], new_act_sample], axis=-1))
        target_q = ops.minimum(target_q_1, target_q_2)
        return log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict], action: Tensor):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            observation: The original observation.
            action: The selected actions.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(ops.cat([outputs_critic_1['state'], action], axis=-1))
        q_2 = self.critic_2(ops.cat([outputs_critic_2['state'], action], axis=-1))
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.trainable_params(), self.target_critic_1_representation.trainable_params()):
            tp.assign_value(tau * ep.data + (1 - tau) * tp.data)
        for ep, tp in zip(self.critic_2_representation.trainable_params(), self.target_critic_2_representation.trainable_params()):
            tp.assign_value(tau * ep.data + (1 - tau) * tp.data)
        for ep, tp in zip(self.critic_1.trainable_params(), self.target_critic_1.trainable_params()):
            tp.assign_value(tau * ep.data + (1 - tau) * tp.data)
        for ep, tp in zip(self.critic_2.trainable_params(), self.target_critic_2.trainable_params()):
            tp.assign_value(tau * ep.data + (1 - tau) * tp.data)
