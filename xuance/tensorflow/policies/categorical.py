import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Union
from xuance.tensorflow import tf, tk, Module
from .core import CategoricalActorNet as ActorNet
from .core import CategoricalActorNet_SAC as Actor_SAC
from .core import CriticNet, BasicQhead


class ActorPolicy(Module):
    """
    Actor for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation)
        else:
            self.representation = representation
            self.representation_info_shape = self.representation.output_shapes
            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)

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
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.representation = representation
                self.representation.build((None,) + self.representation.input_shapes)
                self.representation_info_shape = self.representation.output_shapes
                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation)
                self.critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)
        else:
            self.representation = representation
            self.representation_info_shape = self.representation.output_shapes
            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
            self.critic = CriticNet(representation.output_shapes['state'][0],
                                    critic_hidden_size, normalize, initialize, activation)

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
    """
    Actor-Critic for PPG with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n

        self.use_distributed_training = use_distributed_training
        if self.use_distributed_training:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            with self.mirrored_strategy.scope():
                self.actor_representation = representation
                self.critic_representation = deepcopy(representation)
                self.aux_critic_representation = deepcopy(representation)
                self.representation_info_shape = self.actor_representation.output_shapes
                self.actor_representation.build((None,) + self.actor_representation.input_shapes)
                self.critic_representation.build((None,) + self.critic_representation.input_shapes)
                self.aux_critic_representation.build((None,) + self.aux_critic_representation.input_shapes)

                self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                      actor_hidden_size, normalize, initialize, activation)
                self.critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)
                self.aux_critic = CriticNet(representation.output_shapes['state'][0],
                                            critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.critic_representation = deepcopy(representation)
            self.aux_critic_representation = deepcopy(representation)
            self.representation_info_shape = self.actor_representation.output_shapes

            self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
            self.critic = CriticNet(representation.output_shapes['state'][0],
                                    critic_hidden_size, normalize, initialize, activation)
            self.aux_critic = CriticNet(representation.output_shapes['state'][0],
                                        critic_hidden_size, normalize, initialize, activation)

    def call(self, observation: Union[np.ndarray, dict], **kwargs):
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
        a_logits = self.actor(policy_outputs['state'])
        value = self.critic(critic_outputs['state'])
        aux_value = self.aux_critic(aux_critic_outputs['state'])
        return policy_outputs, a_logits, value[:, 0], aux_value[:, 0]


class SACDISPolicy(Module):
    """
    Actor-Critic for SAC with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[tk.layers.Layer]): The layer normalization over a minibatch of inputs.
        initialize (Optional[tk.initializers.Initializer]): The parameters initializer.
        activation (Optional[tk.layers.Layer]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False):
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation_info_shape = representation.output_shapes

        self.use_distributed_training = use_distributed_training
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

                self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim,
                                       actor_hidden_size, normalize, initialize, activation)
                self.critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                           critic_hidden_size, normalize, initialize, activation)
                self.critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                           critic_hidden_size, normalize, initialize, activation)
                self.target_critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                                  critic_hidden_size, normalize, initialize, activation)
                self.target_critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                                  critic_hidden_size, normalize, initialize, activation)
        else:
            self.actor_representation = representation
            self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim,
                                   actor_hidden_size, normalize, initialize, activation)

            self.critic_1_representation = deepcopy(representation)
            self.critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                       critic_hidden_size, normalize, initialize, activation)
            self.critic_2_representation = deepcopy(representation)
            self.critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                       critic_hidden_size, normalize, initialize, activation)
            self.target_critic_1_representation = deepcopy(self.critic_1_representation)
            self.target_critic_2_representation = deepcopy(self.critic_2_representation)
            self.target_critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                              critic_hidden_size, normalize, initialize, activation)
            self.target_critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim,
                                              critic_hidden_size, normalize, initialize, activation)
        for ep, tp in zip(self.critic_1.variables, self.target_critic_1.variables):
            tp.assign(ep)
        for ep, tp in zip(self.critic_2.variables, self.target_critic_2.variables):
            tp.assign(ep)

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
        _ = self.actor(outputs['state'])
        act_samples = self.actor.dist.stochastic_sample()
        return outputs, act_samples

    @tf.function
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

        act_prob = self.actor(outputs_actor['state'])
        z = act_prob == 0.0
        z = tf.cast(z, dtype=tf.float32) * 1e-8
        log_action_prob = tf.math.log(act_prob + z)

        q_1 = self.critic_1(outputs_critic_1['state'])
        q_2 = self.critic_2(outputs_critic_2['state'])
        return act_prob, log_action_prob, q_1, q_2

    @tf.function
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

        new_act_prob = self.actor(outputs_actor['state'])
        z = new_act_prob == 0.0
        z = tf.cast(z, dtype=tf.float32) * 1e-8
        log_action_prob = tf.math.log(new_act_prob + z)

        target_q_1 = self.target_critic_1(outputs_critic_1['state'])
        target_q_2 = self.target_critic_2(outputs_critic_2['state'])
        target_q = tf.math.minimum(target_q_1, target_q_2)
        return new_act_prob, log_action_prob, target_q

    @tf.function
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
