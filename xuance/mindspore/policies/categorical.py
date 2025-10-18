from copy import deepcopy
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import Module, Tensor, ops
from xuance.mindspore.utils import ModuleType
from .core import CategoricalActorNet as ActorNet
from .core import CategoricalActorNet_SAC as Actor_SAC
from .core import BasicQhead, CriticNet


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
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False):
        super(ActorPolicy, self).__init__()
        self.is_continuous = False
        self.action_dim = action_space.n

        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    def construct(self, observation: Tensor):
        """
        Returns the hidden states, action distribution.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            logits: The categorical logits for the actor distribution.
        """
        outputs = self.representation(observation)
        logits = self.actor(outputs)
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
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.is_continuous = False
        self.action_dim = action_space.n

        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    def construct(self, observation: Tensor):
        """
        Returns the hidden states, action distribution, and values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            logits: The categorical logits for the actor distribution.
            value: The state values output by critic.
        """
        outputs = self.representation(observation)
        logits = self.actor(outputs)
        value = self.critic(outputs)[:, 0]
        return outputs, logits, value


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
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.is_continuous = False
        self.action_dim = action_space.n

        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.aux_critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation)

    def construct(self, observation: Tensor):
        """
        Returns the actors representation output, action distribution, values, and auxiliary values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            policy_outputs: The outputs of actor representation.
            a_logits: The categorical logits for the actor distribution.
            value: The state values output by critic.
            aux_value: The auxiliary values output by aux_critic.
        """
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        aux_critic_outputs = self.aux_critic_representation(observation)
        a_logits = self.actor(policy_outputs)
        value = self.critic(critic_outputs)[:, 0]
        aux_value = self.aux_critic(aux_critic_outputs)[:, 0]
        return policy_outputs, a_logits, value, aux_value


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
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """
    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False):
        super(SACDISPolicy, self).__init__()
        self.is_continuous = False
        self.action_space = action_space
        self.action_dim = action_space.n
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                               normalize, initialize, activation)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.actor_parameters = self.actor_representation.trainable_params() + self.actor.trainable_params()
        self.critic_parameters = self.critic_1_representation.trainable_params() + self.critic_1.trainable_params() + \
                                 self.critic_2_representation.trainable_params() + self.critic_2.trainable_params()

    def construct(self, observation: Tensor):
        """
        Returns the output of actor representation and samples of actions.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            outputs: The outputs of the actor representation.
            logits: The distribution of actions output by actor.
        """
        outputs = self.actor_representation(observation)
        logits = self.actor(outputs)
        return outputs, logits

    def Qpolicy(self, observation: Union[Tensor, dict]):
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

        logits = self.actor(outputs_actor)
        act_prob = ops.softmax(logits, axis=-1)
        z = act_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = ops.log(act_prob + z)

        q_1 = self.critic_1(outputs_critic_1)
        q_2 = self.critic_2(outputs_critic_2)
        return act_prob, log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[Tensor, dict]):
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

        new_logits = self.actor(outputs_actor)
        new_act_prob = ops.softmax(new_logits, axis=-1)
        z = new_act_prob == 0.0
        z = z.float() * 1e-8  # avoid log(0)
        log_action_prob = ops.log(new_act_prob + z)

        target_q_1 = self.target_critic_1(outputs_critic_1)
        target_q_2 = self.target_critic_2(outputs_critic_2)
        target_q = ops.minimum(target_q_1, target_q_2)
        return new_act_prob, log_action_prob, target_q

    def Qaction(self, observation: Union[Tensor, dict]):
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
        q_1 = self.critic_1(outputs_critic_1)
        q_2 = self.critic_2(outputs_critic_2)
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.trainable_params(), self.target_critic_1_representation.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_2_representation.trainable_params(), self.target_critic_2_representation.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_1.trainable_params(), self.target_critic_1.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_2.trainable_params(), self.target_critic_2.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
