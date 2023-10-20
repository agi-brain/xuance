import copy

import numpy as np
import torch

from xuance.torch.policies import *
from xuance.torch.utils import *
from xuance.torch.representations import Basic_Identical
from gymnasium.spaces.box import Box as Box_pettingzoo
from gymnasium import spaces as spaces_pettingzoo


class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class BasicQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            evalQ_detach = evalQ.clone().detach()
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return rnn_hidden, self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class MFQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(MFQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def sample_actions(self, logits: torch.Tensor):
        dist = Categorical(logits=logits)
        return dist.sample()

    def target_Q(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.target_representation(observation)
        q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.eval_Qtot = mixer
        self.target_Qtot = copy.deepcopy(self.eval_Qtot)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            evalQ_detach = evalQ.clone().detach()
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)

        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return rnn_hidden, self.target_Qhead(q_inputs)

    def Q_tot(self, q, states=None):
        return self.eval_Qtot(q, states)

    def target_Q_tot(self, q, states=None):
        return self.target_Qtot(q, states)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, device, **kwargs)
        self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
        self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
        self.q_feedforward = ff_mixer
        self.target_q_feedforward = copy.deepcopy(self.q_feedforward)

    def q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
        else:
            outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.eval_Qhead_centralized(q_inputs)

    def target_q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
        else:
            outputs = self.target_representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead_centralized(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead_centralized.parameters(), self.target_Qhead_centralized.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.q_feedforward.parameters(), self.target_q_feedforward.parameters()):
            tp.data.copy_(ep)


class Qtran_MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: nn.Module,
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = copy.deepcopy(qtran_mixer)
        self.q_tot = mixer

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.target_representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return outputs, self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.qtran_net.parameters(), self.target_qtran_net.parameters()):
            tp.data.copy_(ep)


class DCG_policy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: nn.Module,
                 utility: Optional[nn.Module] = None,
                 payoffs: Optional[nn.Module] = None,
                 dcgraph: Optional[nn.Module] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(DCG_policy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.utility = utility
        self.target_utility = copy.deepcopy(self.utility)
        self.payoffs = payoffs
        self.target_payoffs = copy.deepcopy(self.payoffs)
        self.graph = dcgraph
        self.dcg_s = False
        if hidden_size_bias is not None:
            self.dcg_s = True
            self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                   normalize, initialize, activation, device)
            self.target_bias = copy.deepcopy(self.bias)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            evalQ_detach = evalQ.clone().detach()
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return rnn_hidden, argmax_action, evalQ

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.utility.parameters(), self.target_utility.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.payoffs.parameters(), self.target_payoffs.parameters()):
            tp.data.copy_(ep)
        if self.dcg_s:
            for ep, tp in zip(self.bias.parameters(), self.target_bias.parameters()):
                tp.data.copy_(ep)


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_space: spaces_pettingzoo,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        action_dim = action_space.shape[0]
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Sigmoid, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class CriticNet(nn.Module):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class Basic_DDPG_policy(nn.Module):
    def __init__(self,
                 action_space: spaces_pettingzoo,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_DDPG_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, action_space,
                                  actor_hidden_size, normalize, initialize, activation, device)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor_net.parameters())
        self.parameters_critic = self.critic_net.parameters()

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        act = self.actor_net(actor_in)
        return outputs, act

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_actor_net(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MADDPG_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: spaces_pettingzoo,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(MADDPG_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_critic = self.critic_net.parameters()

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)


class MATD3_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(MATD3_policy, self).__init__(action_space, n_agents, representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, device)
        self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation, device)
        self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_net_A = copy.deepcopy(self.critic_net_A)
        self.target_critic_net_B = copy.deepcopy(self.critic_net_B)
        # self.parameters_critic = self.critic_net.parameters()

    def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, (qa + qb) / 2.0

    def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        # noise = torch.randn_like(actions).clamp(-1, 1) * 0.1
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        # noise = noise.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        # actions_n = (actions_n + noise).clamp(-1, 1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.target_critic_net_A(critic_in)
        qb = self.target_critic_net_B(critic_in)
        min_q = torch.minimum(qa, qb)
        return outputs_n, min_q

    def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, torch.cat((qa, qb), dim=-1)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net_A.parameters(), self.target_critic_net_A.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net_B.parameters(), self.target_critic_net_B.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
