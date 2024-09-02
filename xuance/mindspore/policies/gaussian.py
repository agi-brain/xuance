from copy import deepcopy
from gym.spaces import Box
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import Module, Tensor
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

    def construct(self, observation: Tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


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
        v = self.critic(outputs['state'])
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
        value = self.critic(critic_outputs['state'])
        aux_value = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a_dist, value, aux_value


class SACPolicy(Module):
    def __init__(self,
                 action_space: Box,
                 representation: Module,
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