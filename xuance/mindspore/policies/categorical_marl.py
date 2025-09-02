import mindspore as ms
import mindspore.nn as nn
from copy import deepcopy
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.mindspore.policies import CategoricalActorNet, ActorNet
from xuance.mindspore.policies.core import CriticNet
from xuance.mindspore.policies import VDN_mixer
from xuance.mindspore.utils import ModuleType, CategoricalDistribution
from xuance.mindspore import Tensor, Module, ModuleDict


class MAAC_Policy(nn.Cell):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with categorical policies.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The discrete action space.
        n_agents (int): The number of agents.
        representation_actor (dict): A dict of representation modules for each agent's actor.
        representation_critic (dict): A dict of representation modules for each agent's critic.
        mixer (Module): The mixer module that mix together the individual values to the total value.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initializer (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation_actor: Dict[str, Module],
                 representation_critic: Dict[str, Module],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initializer: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.is_continuous = False
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic

        self.dim_input_critic, self.n_actions = {}, {}
        self.actor, self.critic = {}, {}
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initializer, activation)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initializer, activation)

        self.mixer = mixer
        self.mixer = mixer
        self._concat = ms.ops.Concat(axis=-1)
        self.expand_dims = ms.ops.ExpandDims()
        self._softmax = nn.Softmax(axis=-1)

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor network and critic networks.

        Parameters:
            dim_action: The dimension of actions.
            dim_actor_rep: The dimension of the output of actor representation.
            dim_critic_rep: The dimension of the output of critic representation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = self._concat([outputs, agent_ids])
        act_logits = self.actor(actor_input)
        if avail_actions is not None:
            act_logits[avail_actions == 0] = -1e10
            act_probs = self._softmax(act_logits)
        else:
            act_probs = self._softmax(act_logits)
        return rnn_hidden, act_probs

    def get_values(self, critic_in: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        shape_obs = critic_in.shape
        # get representation features
        if self.use_rnn:
            batch_size, n_agent, episode_length, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, episode_length, dim_obs), *rnn_hidden)
            outputs = outputs.view(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
            outputs = outputs.view(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = self._concat([outputs, agent_ids])
        v = self.critic(critic_in)
        return rnn_hidden, v

    def value_tot(self, values_n: ms.Tensor, global_state=None):
        if global_state is not None:
            global_state = global_state
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAAC_Policy_Share(MAAC_Policy):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Cell,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self._concat = ms.ops.Concat(axis=-1)
        self.expand_dims = ms.ops.ExpandDims()
        self._softmax = nn.Softmax(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None, state=None):
        batch_size = len(observation)
        if self.use_rnn:
            sequence_length = observation.shape[1]
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
            representated_state = outputs.view(batch_size, self.n_agents, sequence_length, -1)
            actor_critic_input = self._concat([representated_state, agent_ids])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
            actor_critic_input = self._concat([outputs, agent_ids])
        act_logits = self.actor(actor_critic_input)
        if avail_actions is not None:
            act_logits[avail_actions == 0] = -1e10
            act_probs = self._softmax(act_logits)
        else:
            act_probs = self._softmax(act_logits)

        values_independent = self.critic(actor_critic_input)
        if self.use_rnn:
            if self.mixer is None:
                values_tot = values_independent
            else:
                sequence_length = observation.shape[1]
                values_independent = values_independent.transpose(1, 2).reshape(batch_size * sequence_length,
                                                                                self.n_agents)
                values_tot = self.value_tot(values_independent, global_state=state)
                values_tot = values_tot.reshape([batch_size, sequence_length, 1])
                values_tot = values_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        else:
            values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                      global_state=state)
            values_tot = ms.ops.broadcast_to(values_tot.unsqueeze(1), (-1, self.n_agents, -1))

        return rnn_hidden, act_probs, values_tot

    def value_tot(self, values_n: ms.Tensor, global_state=None):
        if global_state is not None:
            global_state = ms.Tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class COMA_Policy(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(COMA_Policy, self).__init__()
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
        critic_input_dim = self.representation.input_shape[0] + self.action_dim * self.n_agents
        if kwargs["use_global_state"]:
            critic_input_dim += kwargs["dim_state"]
        self.critic = COMA_Critic(critic_input_dim, self.action_dim, critic_hidden_size,
                                  normalize, initialize, activation)
        self.target_critic = deepcopy(self.critic)
        self.parameters_critic = self.critic.trainable_params()
        self.parameters_actor = self.representation.trainable_params() + self.actor.trainable_params()
        self.eye = ms.ops.Eye()
        self._softmax = nn.Softmax(axis=-1)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None, epsilon=0.0):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = self._concat([outputs, agent_ids])
        act_logits = self.actor(actor_input)
        act_probs = self._softmax(act_logits)
        act_probs = (1 - epsilon) * act_probs + epsilon * 1 / self.action_dim
        if avail_actions is not None:
            act_probs[avail_actions == 0] = 0.0
        return rnn_hidden, act_probs

    def get_values(self, critic_in: ms.Tensor, *rnn_hidden: ms.Tensor, target=False):
        # get critic values
        v = self.target_critic(critic_in) if target else self.critic(critic_in)
        return [None, None], v

    def copy_target(self):
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value(ep)


class MeanFieldActorCriticPolicy(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
        self.critic = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                critic_hidden_size, normalize, initialize, activation)
        self.parameters_actor = self.actor.trainable_params() + self.representation.trainable_params()
        self.parameters_critic = self.critic.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        input_actor = self._concat([outputs, agent_ids])
        act_dist = self.actor(input_actor)
        return outputs, act_dist

    def get_values(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs, actions_mean, agent_ids])
        return self.critic(critic_in)
