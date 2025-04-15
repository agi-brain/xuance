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
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: ModuleDict,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation[0]
        self.representation_critic = representation[1]
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self._concat = ms.ops.Concat(axis=-1)
        self.expand_dims = ms.ops.ExpandDims()
        self._softmax = nn.Softmax(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = self._concat([outputs['state'], agent_ids])
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
            outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            batch_size, n_agent, dim_obs = tuple(shape_obs)
            outputs = self.representation_critic(critic_in.reshape(-1, dim_obs))
            outputs['state'] = outputs['state'].view(batch_size, n_agent, -1)
            rnn_hidden = None
        # get critic values
        critic_in = self._concat([outputs['state'], agent_ids])
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
            representated_state = outputs['state'].view(batch_size, self.n_agents, sequence_length, -1)
            actor_critic_input = self._concat([representated_state, agent_ids])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
            actor_critic_input = self._concat([outputs['state'], agent_ids])
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
        actor_input = self._concat([outputs['state'], agent_ids])
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
        input_actor = self._concat([outputs['state'], agent_ids])
        act_dist = self.actor(input_actor)
        return outputs, act_dist

    def get_values(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions_mean, agent_ids])
        return self.critic(critic_in)
