import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Union
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.policies.core import CategoricalActorNet as ActorNet
from xuance.tensorflow.policies.core import CategoricalActorNet_SAC as Actor_SAC
from xuance.tensorflow.policies.core import CriticNet
from xuance.tensorflow.policies.core import BasicQhead


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
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        """
        Returns the hidden states, action distribution.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
        """
        outputs = self.representation(observation)
        logits = self.actor(outputs['state'])
        return outputs, logits, None


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
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        logits = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        return outputs, logits, value[:, 0]


class PPGActorCritic(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        assert isinstance(action_space, Discrete)
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.aux_critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initializer, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initializer, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initializer, activation, device)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        aux_critic_outputs = self.aux_critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(aux_critic_outputs['state'])
        return policy_outputs, a, v, aux_v


class SACDISPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_critic = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes

        self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                                     normalize, initializer, activation, device)
        self.critic = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                       initializer, activation, device)
        self.target_representation_critic = deepcopy(self.representation_critic)
        self.target_critic = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                              critic_hidden_size, initializer, activation, device)
        self.target_critic.set_weights(self.critic.get_weights())

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        outputs = self.representation(observation)
        act_prob, act_distribution = self.actor(outputs['state'])
        return outputs, act_prob, act_distribution

    @tf.function
    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.representation(observation)
        outputs_critic = self.target_representation_critic(observation)
        act_prob, act_distribution = self.actor(outputs_actor['state'])
        value = self.target_critic(outputs_critic['state'])
        log_action_prob = tf.math.log(act_prob + 1e-5)
        return act_prob, log_action_prob, value

    @tf.function
    def Qaction(self, observation: Union[np.ndarray, dict]):
        outputs_critic = self.representation_critic(observation)
        return outputs_critic, self.critic(outputs_critic['state'])

    @tf.function
    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs_actor = self.representation(observation)
        outputs_critic = self.representation_critic(observation)
        act_prob, act_distribution = self.actor(outputs_actor['state'])
        # z = act_prob == 0.0
        # z = z.float() * 1e-8
        log_action_prob = tf.math.log(act_prob + 1e-5)
        return act_prob, log_action_prob, self.critic(outputs_critic['state'])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation_critic.variables, self.target_representation_critic.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic.variables, self.target_critic.variables):
            tp.assign((1 - tau) * tp + tau * ep)
