import numpy as np
from gymnasium.spaces import Space
from copy import deepcopy
from xuance.common import Sequence, Optional, Union
from xuance.tensorflow import tf, tk, Module, Tensor
from xuance.tensorflow.representations import Basic_Identical
from .core import GaussianActorNet as ActorNet
from .core import CriticNet, GaussianActorNet_SAC


class ActorPolicy(Module):
    """
    Actor for stochastic policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Space): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 fixed_std: bool = True,
                 use_distributed_training: bool = False):
        super(ActorPolicy, self).__init__()
        self.is_continuous = True
        self.action_dim = action_space.shape[0]

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.representation.build((None,) + self.representation.input_shapes)
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, activation_action)
        else:
            self.representation = representation
            self.representation_info_shape = self.representation.output_shapes
            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, activation_action)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        """
        Returns the hidden states, action distribution.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_mean: The distribution of actions output by actor.
        """
        outputs = self.representation(observation)
        a_mean, a_std = self.actor(outputs['state'])
        return outputs, a_mean, a_std, None


class ActorCriticPolicy(Module):
    """
    Actor-Critic for stochastic policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Space): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.is_continuous = True
        self.action_dim = action_space.shape[0]

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.representation_info_shape = self.representation.output_shapes
                self.representation.build((None,) + self.representation.input_shapes)
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, activation_action)
                self.critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)
        else:
            self.representation = representation
            self.representation_info_shape = self.representation.output_shapes
            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, activation_action)
            self.critic = CriticNet(representation.output_shapes['state'][0],
                                    critic_hidden_size, normalize, initialize, activation)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        """
        Returns the hidden states, action distribution, and values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_mean: The mean variable of the gaussian distribution.
            value: The state values output by critic.
        """
        outputs = self.representation(observation)
        a_mean, a_std = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a_mean, a_std, v[:, 0]


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
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.is_continuous = True
        self.action_dim = action_space.shape[0]

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.actor_representation = representation
                self.critic_representation = deepcopy(representation)
                self.representation_info_shape = self.actor_representation.output_shapes
                self.actor_representation.build((None,) + self.actor_representation.input_shapes)
                self.critic_representation.build((None,) + self.critic_representation.input_shapes)

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, activation_action)
                self.critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)
                self.aux_critic = CriticNet(representation.output_shapes['state'][0],
                                            critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.critic_representation = deepcopy(representation)
            self.representation_info_shape = self.actor_representation.output_shapes
            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, activation_action)
            self.critic = CriticNet(representation.output_shapes['state'][0],
                                    critic_hidden_size, normalize, initialize, activation)
            self.aux_critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        """
        Returns the actors representation output, action distribution, values, and auxiliary values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            policy_outputs: The outputs of actor representation.
            a_mean: The distribution of actions output by actor.
            value: The state values output by critic.
            aux_value: The auxiliary values output by aux_critic.
        """
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        a_mean, a_std = self.actor(policy_outputs['state'])
        value = self.critic(critic_outputs['state'])
        aux_value = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a_mean, a_std, value[:, 0], aux_value[:, 0]


class SACPolicy(Module):
    """
    Actor-Critic for SAC with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Space): The continuous action space.
        representation (Basic_Identical): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[Module]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[Module] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(SACPolicy, self).__init__()
        self.is_continuous = True
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.use_distributed_training = use_distributed_training
        self.activation_action = activation_action
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.actor_representation = representation
                self.critic_1_representation = deepcopy(representation)
                self.critic_2_representation = deepcopy(representation)
                self.target_critic_1_representation = deepcopy(self.critic_1_representation)
                self.target_critic_2_representation = deepcopy(self.critic_2_representation)
                self.actor_representation.build((None,) + self.actor_representation.input_shapes)
                self.critic_1_representation.build((None,) + self.critic_1_representation.input_shapes)
                self.critic_2_representation.build((None,) + self.critic_2_representation.input_shapes)

                self.actor = GaussianActorNet_SAC(representation.output_shapes['state'][0], self.action_dim,
                                                  actor_hidden_size, normalize, initialize,
                                                  activation, activation_action)
                self.critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                          critic_hidden_size, normalize, initialize, activation)
                self.critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                          critic_hidden_size, normalize, initialize, activation)
                self.target_critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                                 critic_hidden_size, normalize, initialize, activation)
                self.target_critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                                 critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.actor = GaussianActorNet_SAC(representation.output_shapes['state'][0], self.action_dim,
                                              actor_hidden_size, normalize, initialize,
                                              activation, activation_action)

            self.critic_1_representation = deepcopy(representation)
            self.critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation)
            self.critic_2_representation = deepcopy(representation)
            self.critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation)
            self.target_critic_1_representation = deepcopy(self.critic_1_representation)
            self.target_critic_2_representation = deepcopy(self.critic_2_representation)
            self.target_critic_1 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                             critic_hidden_size, normalize, initialize, activation)
            self.target_critic_2 = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                             critic_hidden_size, normalize, initialize, activation)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

    @property
    def actor_trainable_variables(self):
        return self.actor_representation.trainable_variables + self.actor.trainable_variables

    @property
    def critic_trainable_variables(self):
        return self.critic_1_representation.trainable_variables + self.critic_1.trainable_variables + \
               self.critic_2_representation.trainable_variables + self.critic_2.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
        """
        Returns the output of actor representation and samples of actions.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            outputs: The outputs of the actor representation.
            act_sample: The sampled actions from the distribution output by the actor.
        """
        outputs = self.actor_representation(observation)
        a_mean, a_std = self.actor(outputs['state'])
        eps = tf.random.normal(shape=tf.shape(a_mean))  # 𝜖 ~ N(0, 1)
        action_sampled = a_mean + a_std * eps  # Reparameterization trick
        actions_activated = self.activation_action(action_sampled)
        # calculate log prob
        log_std = tf.math.log(a_std + 1e-8)
        log_prob = -0.5 * (((action_sampled - a_mean) / (a_std + 1e-8)) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
        correction = - 2. * (tf.math.log(2.0) - action_sampled - tk.activations.softplus(-2. * action_sampled))
        log_prob += correction
        log_action_prob = tf.reduce_sum(log_prob, axis=-1)
        return outputs, actions_activated, log_action_prob

    @tf.function
    def Qpolicy(self, observation: Union[np.ndarray, dict], actions: Union[np.ndarray, dict]):
        """
        Feedforward and calculate the log of action probabilities, and Q-values.

        Parameters:
            observation (Union[np.ndarray, dict]): The original observation of an agent.
            actions (Union[np.ndarray, dict]): The actions.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        critic_1_in = tf.concat([outputs_critic_1['state'], actions], axis=-1)
        critic_2_in = tf.concat([outputs_critic_2['state'], actions], axis=-1)

        q_1 = self.critic_1(critic_1_in)
        q_2 = self.critic_2(critic_2_in)
        return q_1[:, 0], q_2[:, 0]

    @tf.function
    def Qtarget(self, next_observation: Union[np.ndarray, dict], next_actions: Union[np.ndarray, dict]):
        """
        Calculate the log of action probabilities and Q-values with target networks.

        Parameters:
            next_observation (Union[np.ndarray, dict]): The observations of next step.
            next_actions (Union[np.ndarray, dict]): The actions of next step.

        Returns:
            target_q: The minimum of Q-values calculated by the target critic networks.
        """
        outputs_critic_1 = self.target_critic_1_representation(next_observation)
        outputs_critic_2 = self.target_critic_2_representation(next_observation)

        critic_1_in = tf.concat([outputs_critic_1['state'], next_actions], axis=-1)
        critic_2_in = tf.concat([outputs_critic_2['state'], next_actions], axis=-1)

        target_q_1 = self.target_critic_1(critic_1_in)
        target_q_2 = self.target_critic_2(critic_2_in)
        target_q = tf.math.minimum(target_q_1, target_q_2)
        return target_q[:, 0]

    @tf.function
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.variables, self.target_critic_1_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_2_representation.variables, self.target_critic_2_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_1.variables, self.target_critic_1.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_2.variables, self.target_critic_2.variables):
            tp.assign((1 - tau) * tp + tau * ep)
