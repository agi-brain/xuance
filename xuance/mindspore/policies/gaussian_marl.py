import mindspore as ms
import numpy as np
from copy import deepcopy
from gym.spaces import Box
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.mindspore.utils import ModuleType
from xuance.mindspore import Tensor, Module, ModuleDict
from .core import GaussianActorNet, GaussianActorNet_SAC, CriticNet


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with Gaussian policies
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation[0]
        self.representation_critic = representation[1]
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.actor = GaussianActorNet(self.representation.output_shapes['state'][0], n_agents, self.action_dim,
                              actor_hidden_size, normalize, initialize, activation)
        dim_input_critic = self.representation_critic.output_shapes['state'][0]
        self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor,
                  *rnn_hidden: ms.tensor, **kwargs):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor(actor_input)
        return rnn_hidden, mu_a

    def get_values(self, critic_in: ms.tensor, agent_ids: ms.tensor, *rnn_hidden: ms.tensor, **kwargs):
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

    def value_tot(self, values_n: ms.tensor, global_state=None):
        if global_state is not None:
            global_state = ms.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class Basic_ISAC_Policy(Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(Basic_ISAC_Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = GaussianActorNet_SAC(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim
        self.critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = GaussianActorNet_SAC(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initialize, activation)
        self.target_critic_net = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                           normalize, initialize, activation)
        self.parameters_actor = list(self.representation.trainable_params()) + list(self.actor_net.trainable_params())
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self.soft_update(tau=1.0)

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor_net(actor_in)
        return outputs, mu_a

    def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.target_actor_net(actor_in)
        return mu_a

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class MASAC_policy(Basic_ISAC_Policy):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(MASAC_policy, self).__init__()

    def construct(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.actor_net(actor_in)
        return outputs, mu_a

    def critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.critic_net(critic_in)

    def critic_for_train(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.tensor, actions: ms.tensor, agent_ids: ms.tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to_act(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.tensor, agent_ids: ms.tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        mu_a = self.target_actor_net(actor_in)
        return mu_a

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
