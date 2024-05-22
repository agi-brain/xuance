import copy

from xuance.torch.policies import *
from xuance.torch.utils import *
from torch.distributions import Categorical


class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
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
        layers_.extend(mlp_block(input_shape[0], n_actions, None, None, None, device)[0])
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
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.n_actions, n_agents,
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
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.n_actions, self.n_actions,
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
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.n_actions, n_agents,
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.n_actions = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.n_actions, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = copy.deepcopy(qtran_mixer)
        self.q_tot = mixer

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
        return rnn_hidden, outputs['state'], argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor, *rnn_hidden: torch.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return rnn_hidden, outputs['state'], self.target_Qhead(q_inputs)

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
        self.n_actions = action_space.n
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
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class Basic_DDPG_policy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 ):
        super(Basic_DDPG_policy, self).__init__()
        self.action_dim = action_space.shape[-1]
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = representation.output_shapes['state'][0] + self.action_dim

        self.actor_representation = representation
        self.actor = ActorNet(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic_representation = copy.deepcopy(representation)
        self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)

        self.target_actor_representation = copy.deepcopy(self.actor_representation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_representation = copy.deepcopy(self.critic_representation)
        self.target_critic = copy.deepcopy(self.critic)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.actor_representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        act = self.actor(actor_in)
        return outputs, act

    def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.critic_representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.critic(critic_in)

    def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.target_critic_representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.target_critic(critic_in)

    def Atarget(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.target_actor_representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_actor(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MADDPG_policy(Basic_DDPG_policy, nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        nn.Module.__init__(self)
        self.action_dim = action_space.shape[-1]
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents

        self.actor_representation = representation
        self.actor = ActorNet(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic_representation = copy.deepcopy(representation)
        self.critic = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                normalize, initialize, activation, device)

        self.target_actor_representation = copy.deepcopy(self.actor_representation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_representation = copy.deepcopy(self.critic_representation)
        self.target_critic = copy.deepcopy(self.critic)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs = self.critic_representation(observation)
        critic_in = torch.concat([outputs['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                  actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                  agent_ids], dim=-1)
        return self.critic(critic_in)

    def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs = self.target_critic_representation(observation)
        critic_in = torch.concat([outputs['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                  actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                  agent_ids], dim=-1)
        return self.target_critic(critic_in)


class MATD3_policy(Basic_DDPG_policy, nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: nn.Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        nn.Module.__init__(self)
        self.action_dim = action_space.shape[-1]
        self.n_agents = n_agents
        self.representation_info_shape = representation.output_shapes
        dim_input_actor = representation.output_shapes['state'][0]
        dim_input_critic = (representation.output_shapes['state'][0] + self.action_dim) * self.n_agents

        self.actor_representation = representation
        self.actor = ActorNet(dim_input_actor, n_agents, self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic_A_representation = copy.deepcopy(representation)
        self.critic_A = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.critic_B_representation = copy.deepcopy(representation)
        self.critic_B = CriticNet(dim_input_critic, n_agents, critic_hidden_size,
                                  normalize, initialize, activation, device)
        self.target_actor_representation = copy.deepcopy(self.actor_representation)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_A_representation = copy.deepcopy(self.critic_A_representation)
        self.target_critic_A = copy.deepcopy(self.critic_A)
        self.target_critic_B_representation = copy.deepcopy(self.critic_B_representation)
        self.target_critic_B = copy.deepcopy(self.critic_B)

        self.parameters_actor = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.parameters_critic = list(self.critic_A_representation.parameters()) + list(
            self.critic_A.parameters()) + list(self.critic_B_representation.parameters()) + list(
            self.critic_B.parameters())

    def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        critic_A_in = torch.concat([outputs_critic_A['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_B_in = torch.concat([outputs_critic_B['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        qa, qb = self.critic_A(critic_A_in), self.critic_B(critic_B_in)
        return (qa + qb) / 2.0

    def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_critic_A = self.target_critic_A_representation(observation)
        outputs_critic_B = self.target_critic_B_representation(observation)
        critic_A_in = torch.concat([outputs_critic_A['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_B_in = torch.concat([outputs_critic_B['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        qa, qb = self.target_critic_A(critic_A_in), self.target_critic_B(critic_B_in)
        min_q = torch.minimum(qa, qb)
        return min_q

    def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_critic_A = self.critic_A_representation(observation)
        outputs_critic_B = self.critic_B_representation(observation)
        critic_A_in = torch.concat([outputs_critic_A['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        critic_B_in = torch.concat([outputs_critic_B['state'].reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    actions.reshape(bs, 1, -1).expand(-1, self.n_agents, -1),
                                    agent_ids], dim=-1)
        qa, qb = self.critic_A(critic_A_in), self.critic_B(critic_B_in)
        return torch.cat((qa, qb), dim=-1)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A_representation.parameters(), self.target_critic_A_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_A.parameters(), self.target_critic_A.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B_representation.parameters(), self.target_critic_B_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_B.parameters(), self.target_critic_B.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
