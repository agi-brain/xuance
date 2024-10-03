import numpy as np
from copy import deepcopy
from gym.spaces import Space, Discrete
from xuance.common import Sequence, Optional, Union, Callable
from xuance.tensorflow import tf, tk, Module, Tensor
from .core import BasicQhead, BasicRecurrent, DuelQhead, C51Qhead, QRDQNhead, ActorNet, CriticNet


class BasicQnetwork(Module):
    """
    The base class to implement DQN based policy

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                             hidden_size, normalize, initialize, activation)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                               hidden_size, normalize, initialize, activation)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        else:
            self.mirrored_strategy = None
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                         hidden_size, normalize, initialize, activation)
            self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                           hidden_size, normalize, initialize, activation)
            self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

    @tf.function
    def call(self, observation: Union[Tensor, np.ndarray]):
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
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    @tf.function
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
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs_target, tf.stop_gradient(argmax_action), tf.stop_gradient(targetQ)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class DuelQnetwork(Module):
    """
    The policy for deep dueling Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(DuelQnetwork, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                            hidden_size, normalize, initialize, activation)
                self.target_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                              hidden_size, normalize, initialize, activation)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        else:
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            self.eval_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                        hidden_size, normalize, initialize, activation)
            self.target_Qhead = DuelQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                          hidden_size, normalize, initialize, activation)
            self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    @tf.function
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
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, targetQ

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class NoisyQnetwork(Module):
    """
    The policy for noisy deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(NoisyQnetwork, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                             hidden_size, normalize, initialize, activation)
                self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                               hidden_size, normalize, initialize, activation)
        else:
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                         hidden_size, normalize, initialize, activation)
            self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim,
                                           hidden_size, normalize, initialize, activation)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.noise_scale = 0.0
        self.eval_noise_parameter = []
        self.target_noise_parameter = []

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

    def update_noise(self, noisy_bound: float = 0.0):
        """Updates the noises for network parameters."""
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Qhead.variables:
            self.eval_noise_parameter.append(
                tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)
            self.target_noise_parameter.append(
                tf.random.uniform(shape=parameter.shape, minval=-1.0, maxval=1.0) * noisy_bound)

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        for parameter, noise_param in zip(self.eval_Qhead.variables, self.eval_noise_parameter):
            parameter.assign_add(noise_param)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    @tf.function
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
        for parameter, noise_param in zip(self.target_Qhead.variables, self.target_noise_parameter):
            parameter.assign_add(noise_param)
        targetQ = self.target_Qhead(outputs['state'])
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, tf.stop_gradient(targetQ)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class C51Qnetwork(Module):
    """
    The policy for C51 distributional deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        atom_num (int): The number of atoms.
        v_min (float): The lower bound of value distribution.
        v_max (float): The upper bound of value distribution.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 atom_num: int,
                 v_min: float,
                 v_max: float,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(C51Qnetwork, self).__init__()
        self.action_dim = action_space.n
        self.atom_num = atom_num
        self.v_min = v_min
        self.v_max = v_max

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim,
                                           self.atom_num, hidden_size, normalize, initialize, activation)
                self.target_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim,
                                             self.atom_num, hidden_size, normalize, initialize, activation)
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())
        else:
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            self.eval_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim,
                                       self.atom_num, hidden_size, normalize, initialize, activation)
            self.target_Zhead = C51Qhead(self.representation.output_shapes['state'][0], self.action_dim,
                                         self.atom_num, hidden_size, normalize, initialize, activation)
            self.target_Zhead.set_weights(self.eval_Zhead.get_weights())

        self.supports = tf.cast(tf.linspace(self.v_min, self.v_max, self.atom_num), dtype=tf.float32)
        self.deltaz = (v_max - v_min) / (atom_num - 1)

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Zhead.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        eval_Q = tf.reduce_sum(self.supports * eval_Z, axis=-1)
        argmax_action = tf.math.argmax(eval_Q, axis=-1)
        return outputs, argmax_action, eval_Z

    @tf.function
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
        target_Q = tf.reduce_sum(self.supports * target_Z, axis=-1)
        argmax_action = tf.math.argmax(target_Q, axis=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


class QRDQN_Network(Module):
    """
    The policy for quantile regression deep Q-networks.

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        quantile_num (int): The number of quantiles.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 quantile_num: int,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(QRDQN_Network, self).__init__()
        self.action_dim = action_space.n
        self.quantile_num = quantile_num

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim,
                                            self.quantile_num, hidden_size, normalize, initialize, activation)
                self.target_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim,
                                              self.quantile_num, hidden_size, normalize, initialize, activation)
                self.target_Zhead.set_weights(self.eval_Zhead.get_weights())
        else:
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            self.eval_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim,
                                        self.quantile_num, hidden_size, normalize, initialize, activation)
            self.target_Zhead = QRDQNhead(self.representation.output_shapes['state'][0], self.action_dim,
                                          self.quantile_num, hidden_size, normalize, initialize, activation)
            self.target_Zhead.set_weights(self.eval_Zhead.get_weights())

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Zhead.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        eval_Q = tf.reduce_mean(eval_Z, axis=-1)
        argmax_action = tf.math.argmax(eval_Q, axis=-1)
        return outputs, argmax_action, eval_Z

    @tf.function
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
        target_Q = tf.reduce_mean(target_Z, axis=-1)
        argmax_action = tf.math.argmax(target_Q, axis=-1)
        return outputs, argmax_action, target_Z

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Zhead.set_weights(self.eval_Zhead.get_weights())


class DDPGPolicy(Module):
    """
    The policy of deep deterministic policy gradient.

    Args:
        action_space (Space): The action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): List of hidden units for actor network.
        critic_hidden_size (Sequence[int]): List of hidden units for critic network.
        normalize (Optional[Module]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
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
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(DDPGPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.actor_representation = representation
                self.critic_representation = deepcopy(representation)
                self.target_actor_representation = deepcopy(self.actor_representation)
                self.target_critic_representation = deepcopy(self.critic_representation)
                self.actor_representation.build((None, ) + self.actor_representation.input_shapes)
                self.critic_representation.build((None, ) + self.critic_representation.input_shapes)

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, activation_action)
                self.critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                        critic_hidden_size, normalize, initialize, activation)
                self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                             actor_hidden_size, normalize, initialize, activation, activation_action)
                self.target_critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                               critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.critic_representation = deepcopy(representation)
            self.target_actor_representation = deepcopy(self.actor_representation)
            self.target_critic_representation = deepcopy(self.critic_representation)

            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, activation_action)
            self.critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
            self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation, activation_action)
            self.target_critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                           critic_hidden_size, normalize, initialize, activation)
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign(ep)
        for ep, tp in zip(self.critic.variables, self.target_critic.variables):
            tp.assign(ep)

    @property
    def actor_trainable_variables(self):
        return self.actor_representation.trainable_variables + self.actor.trainable_variables

    @property
    def critic_trainable_variables(self):
        return self.critic_representation.trainable_variables + self.critic.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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

    @tf.function
    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values via target networks."""
        outputs_actor = self.target_actor_representation(observation)
        outputs_critic = self.target_critic_representation(observation)
        act = self.target_actor(outputs_actor['state'])
        q_ = self.target_critic(tf.concat([outputs_critic['state'], act], axis=-1))
        return q_[:, 0]

    @tf.function
    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        """Returns the evaluated Q-values of state-action pairs."""
        outputs = self.critic_representation(observation)
        q = self.critic(tf.concat([outputs['state'], action], axis=-1))
        return q[:, 0]

    @tf.function
    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values by calculating actions via actor networks."""
        outputs_actor = self.actor_representation(observation)
        act = self.actor(outputs_actor['state'])
        outputs_critic = self.critic_representation(observation)
        q_eval = self.critic(tf.concat([outputs_critic['state'], act], axis=-1))
        return q_eval[:, 0]

    @tf.function
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.variables, self.target_actor_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_representation.variables, self.target_critic_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic.variables, self.target_critic.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class TD3Policy(Module):
    """
    The policy of twin delayed deep deterministic policy gradient.

    Args:
        action_space (Space): The action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): List of hidden units for actor network.
        critic_hidden_size (Sequence[int]): List of hidden units for critic network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Space,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(TD3Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.actor_representation = representation
                self.critic_A_representation = deepcopy(representation)
                self.critic_B_representation = deepcopy(representation)
                self.target_actor_representation = deepcopy(self.actor_representation)
                self.target_critic_A_representation = deepcopy(self.critic_A_representation)
                self.target_critic_B_representation = deepcopy(self.critic_B_representation)
                self.actor_representation.build((None,) + self.actor_representation.input_shapes)
                self.critic_A_representation.build((None,) + self.critic_A_representation.input_shapes)
                self.critic_B_representation.build((None,) + self.critic_B_representation.input_shapes)

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation, activation_action)
                self.critic_A = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                          critic_hidden_size, normalize, initialize, activation)
                self.critic_B = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                          critic_hidden_size, normalize, initialize, activation)
                self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                             actor_hidden_size, normalize, initialize, activation, activation_action)
                self.target_critic_A = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                                 critic_hidden_size, normalize, initialize, activation)
                self.target_critic_B = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                                 critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.critic_A_representation = deepcopy(representation)
            self.critic_B_representation = deepcopy(representation)
            self.target_actor_representation = deepcopy(self.actor_representation)
            self.target_critic_A_representation = deepcopy(self.critic_A_representation)
            self.target_critic_B_representation = deepcopy(self.critic_B_representation)

            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, activation_action)
            self.critic_A = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation)
            self.critic_B = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation)
            self.target_actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation, activation_action)
            self.target_critic_A = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                             critic_hidden_size, normalize, initialize, activation)
            self.target_critic_B = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                             critic_hidden_size, normalize, initialize, activation)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_A.set_weights(self.critic_A.get_weights())
        self.target_critic_B.set_weights(self.critic_B.get_weights())

    @property
    def actor_trainable_variables(self):
        return self.actor_representation.trainable_variables + self.actor.trainable_variables

    @property
    def critic_trainable_variables(self):
        return self.critic_A_representation.trainable_variables + self.critic_A.trainable_variables + \
               self.critic_B_representation.trainable_variables + self.critic_B.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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

    @tf.function
    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values via target networks."""
        outputs_actor = self.target_actor_representation(observation)
        outputs_critic_A = self.target_critic_A_representation(observation)
        outputs_critic_B = self.target_critic_B_representation(observation)
        act = self.target_actor(outputs_actor['state'])
        noise = tf.random.uniform(act.shape, -1, 1) * 0.2
        noise = tf.clip_by_value(noise, -0.5, 0.5)
        act = tf.clip_by_value(act + noise, -1, 1)

        qa = self.target_critic_A(tf.concat([outputs_critic_A['state'], act], axis=-1))
        qb = self.target_critic_B(tf.concat([outputs_critic_B['state'], act], axis=-1))
        min_q = tf.math.minimum(qa, qb)
        return min_q[:, 0]

    @tf.function
    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        """Returns the evaluated Q-values of state-action pairs."""
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        q_eval_a = self.critic_A(tf.concat([outputs_critic_A['state'], action], axis=-1))
        q_eval_b = self.critic_B(tf.concat([outputs_critic_B['state'], action], axis=-1))
        return q_eval_a[:, 0], q_eval_b[:, 0]

    @tf.function
    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """Returns the evaluated Q-values by calculating actions via actor networks."""
        outputs_actor = self.actor_representation(observation)
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        act = self.actor(outputs_actor['state'])
        q_eval_a = tf.expand_dims(self.critic_A(tf.concat([outputs_critic_A['state'], act], axis=-1)), axis=1)
        q_eval_b = tf.expand_dims(self.critic_B(tf.concat([outputs_critic_B['state'], act], axis=-1)), axis=1)
        return (q_eval_a + q_eval_b) / 2.0

    @tf.function
    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.variables, self.target_actor_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_A_representation.variables, self.target_critic_A_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_B_representation.variables, self.target_critic_B_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.actor.variables, self.target_actor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_A.variables, self.target_critic_A.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_B.variables, self.target_critic_B.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class PDQNPolicy(Module):
    """
    The policy of parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(PDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize, initialize, activation)
        self.dim_input = self.observation_space.shape[0] + self.conact_size
        self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, activation_action)
        self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                          qnetwork_hidden_size, normalize, initialize, activation)
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, activation_action)
        self.target_conactor.set_weights(self.conactor.get_weights())
        self.target_qnetwork.set_weights(self.qnetwork.get_weights())

    @tf.function
    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    @tf.function
    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    @tf.function
    def Qtarget(self, state, action):
        input_q = tf.concat((state, action), axis=1)
        target_q = self.target_qnetwork(input_q)
        return target_q

    @tf.function
    def Qeval(self, state, action):
        input_q = tf.concat((state, action), axis=1)
        eval_q = self.qnetwork(input_q)
        return eval_q

    @tf.function
    def Qpolicy(self, state):
        conact = self.conactor(state)
        input_q = tf.concat((state, conact), axis=1)
        policy_q = tf.reduce_sum(self.qnetwork(input_q))
        return policy_q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.representation.variables, self.target_representation.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MPDQNPolicy(Module):
    """
    The policy of multi-pass parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(MPDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.observation_space = observation_space
        self.obs_size = self.observation_space.shape[0]
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                   qnetwork_hidden_size, normalize, initialize, activation)
        self.dim_input = self.observation_space.shape[0] + self.conact_size
        self.qnetwork._set_inputs(tf.TensorSpec([None, self.dim_input], tf.float32, name='inputs'))
        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, activation_action)

        self.target_qnetwork = BasicQhead(self.observation_space.shape[0] + self.conact_size, self.num_disact,
                                          qnetwork_hidden_size, normalize, initialize, activation)
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, activation_action)
        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.soft_update(tau=1.0)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.target_qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, 1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qeval(self, state, action):
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(action)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = action[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, axis=1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qpolicy(self, state):
        conact = self.conactor(state)
        batch_size = state.shape[0]
        Q = []
        input_q = tf.concat((state, tf.zeros_like(conact)), axis=1)
        input_q = tf.tile(input_q, (self.num_disact, 1)).numpy()
        for i in range(self.num_disact):
            input_q[i * batch_size:(i + 1) * batch_size,
            self.obs_size + self.offsets[i]: self.obs_size + self.offsets[i + 1]] \
                = conact[:, self.offsets[i]:self.offsets[i + 1]]
        eval_qall = self.qnetwork(input_q)
        for i in range(self.num_disact):
            eval_q = eval_qall[i * batch_size:(i + 1) * batch_size, i]
            if len(eval_q.shape) == 1:
                eval_q = tf.expand_dims(eval_q, axis=1)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def soft_update(self, tau=0.005):
        # for ep, tp in zip(self.representation.variables, self.target_representation.variables):
        #     tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.qnetwork.variables, self.target_qnetwork.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class SPDQNPolicy(Module):
    """
    The policy of split parameterised deep Q network.

    Args:
        observation_space: The observation spaces.
        action_space: The action spaces.
        representation (Module): The representation module.
        conactor_hidden_size (Sequence[int]): List of hidden units for actor network.
        qnetwork_hidden_size (Sequence[int]): List of hidden units for q network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        activation_action (Optional[tk.layers.Layer]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 representation: Module,
                 conactor_hidden_size: Sequence[int],
                 qnetwork_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(SPDQNPolicy, self).__init__()
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())
        self.qnetwork, self.target_qnetwork = [], []
        for k in range(self.num_disact):
            self.qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                           initialize, activation))
            dim_input = self.observation_space.shape[0] + self.conact_sizes[k]
            self.qnetwork[k]._set_inputs(tf.TensorSpec([None, dim_input], tf.float32, name='inputs_%d' % (k)))

        self.conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                 initialize, activation, activation_action)
        for k in range(self.num_disact):
            self.target_qnetwork.append(
                BasicQhead(self.observation_space.shape[0] + self.conact_sizes[k], 1, qnetwork_hidden_size, normalize,
                           initialize, activation))
        self.target_conactor = ActorNet(self.observation_space.shape[0], self.conact_size, conactor_hidden_size,
                                        initialize, activation, activation_action)

        self.offsets = self.conact_sizes.cumsum()
        self.offsets = np.insert(self.offsets, 0, 0)
        self.soft_update(tau=1.0)

    def Atarget(self, state):
        target_conact = self.target_conactor(state)
        return target_conact

    def con_action(self, state):
        conaction = self.conactor(state)
        return conaction

    def Qtarget(self, state, action):
        target_Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.target_qnetwork[i](input_q)
            target_Q.append(eval_q)
        target_Q = tf.concat(target_Q, axis=1)
        return target_Q

    def Qeval(self, state, action):
        Q = []
        for i in range(self.num_disact):
            conact = action[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def Qpolicy(self, state):
        conacts = self.conactor(state)
        Q = []
        for i in range(self.num_disact):
            conact = conacts[:, self.offsets[i]:self.offsets[i + 1]]
            input_q = tf.concat((state, conact), axis=1)
            eval_q = self.qnetwork[i](input_q)
            Q.append(eval_q)
        Q = tf.concat(Q, axis=1)
        return Q

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.conactor.variables, self.target_conactor.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for qnet, target_qnet in zip(self.qnetwork, self.target_qnetwork):
            for ep, tp in zip(qnet.variables, target_qnet.variables):
                tp.assign((1 - tau) * tp + tau * ep)


class DRQNPolicy(Module):
    """
    The policy of deep recurrent Q-networks.

    Args:
        action_space (Discrete): The action space.
        representation (Module): The representation module.
        **kwargs: The other arguments.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 **kwargs):
        super(DRQNPolicy, self).__init__()
        self.recurrent_layer_N = kwargs['recurrent_layer_N']
        self.rnn_hidden_dim = kwargs['recurrent_hidden_size']
        self.action_dim = action_space.n
        kwargs["action_dim"] = self.action_dim
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.cnn = True if representation.name == "basic_cnn" else False

        self.use_distributed_training = kwargs["use_distributed_training"]
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.target_representation = deepcopy(representation)
                self.representation_info_shape = self.representation.output_shapes
                kwargs["input_dim"] = self.representation.output_shapes['state'][0]
                self.eval_Qhead = BasicRecurrent(**kwargs)
                self.target_Qhead = BasicRecurrent(**kwargs)
                self.representation.build((None,) + self.representation.input_shapes)
                self.target_representation.build((None,) + self.target_representation.input_shapes)
                self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        else:
            self.representation = representation
            self.target_representation = deepcopy(representation)
            self.representation_info_shape = self.representation.output_shapes
            kwargs["input_dim"] = self.representation.output_shapes['state'][0]
            self.eval_Qhead = BasicRecurrent(**kwargs)
            self.target_Qhead = BasicRecurrent(**kwargs)
            self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    @property
    def trainable_variables(self):
        return self.representation.trainable_variables + self.eval_Qhead.trainable_variables

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], *rnn_hidden: Union[Tensor, np.ndarray], **kwargs):
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
            output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            obs_shape = observation.shape
            observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.representation(observations_in)
            output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
        if self.lstm:
            hidden_states, cell_states, evalQ = self.eval_Qhead(output_states)
        else:
            hidden_states, evalQ = self.eval_Qhead(output_states)
            cell_states = None
        argmax_action = tf.math.argmax(evalQ[:, -1], axis=-1)
        return outputs, argmax_action, evalQ, (hidden_states, cell_states)

    @tf.function
    def target(self, observation: Union[np.ndarray, dict], *rnn_hidden: Tensor):
        if self.cnn:
            obs_shape = observation.shape
            outputs = self.target_representation(observation.reshape((-1,) + obs_shape[-3:]))
            output_states = outputs['state'].reshape(obs_shape[0:-3] + (-1,))
        else:
            obs_shape = observation.shape
            observations_in = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.target_representation(observations_in)
            output_states = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation.output_shapes['state'])
        if self.lstm:
            hidden_states, cell_states, targetQ = self.target_Qhead(output_states)
        else:
            hidden_states, targetQ = self.target_Qhead(output_states)
            cell_states = None
        argmax_action = tf.math.argmax(targetQ, axis=-1)
        return outputs, argmax_action, targetQ, (hidden_states, cell_states)

    def init_hidden(self, batch):
        hidden_states = np.zeros(shape=(self.recurrent_layer_N, batch, self.rnn_hidden_dim))
        cell_states = np.zeros_like(hidden_states) if self.lstm else None
        return hidden_states, cell_states

    def init_hidden_item(self, rnn_hidden, i):
        if self.lstm:
            rnn_hidden_0, rnn_hidden_1 = rnn_hidden[0], rnn_hidden[1]
            rnn_hidden_0[i:i + 1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
            rnn_hidden_1[i:i + 1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
            return rnn_hidden_0, rnn_hidden_1
        else:
            rnn_hidden[i:i + 1] = np.zeros((self.recurrent_layer_N, self.rnn_hidden_dim))
            return rnn_hidden

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
