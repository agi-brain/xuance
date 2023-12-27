from xuance.mindspore.policies import *
from xuance.mindspore.utils import *
import copy
from xuance.mindspore.representations import Basic_Identical
from mindspore.nn.probability.distribution import Categorical


class BasicQhead(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers_)

    def construct(self, x: ms.tensor):
        return self.model(x)


class BasicQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = copy.deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(axis=-1)
        else:
            argmax_action = evalQ.argmax(axis=-1)
        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        return rnn_hidden, self.target_Qhead(q_inputs)

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class MFQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(MFQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, hidden_size, normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self._concat = ms.ops.Concat(axis=-1)
        self._dist = Categorical(dtype=ms.float32)

    def construct(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        q_inputs = self._concat([outputs['state'], actions_mean, agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(axis=-1)
        return outputs, argmax_action, evalQ

    def sample_actions(self, logits: ms.Tensor):
        return self._dist.sample(probs=logits).astype(ms.int32)

    def target_Q(self, observation: ms.Tensor, actions_mean: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        q_inputs = self._concat([outputs['state'], actions_mean, agent_ids])
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class MixingQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.eval_Qtot = mixer
        self.target_Qtot = copy.deepcopy(self.eval_Qtot)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = copy.deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(axis=-1)
        else:
            argmax_action = evalQ.argmax(axis=-1)
        return rnn_hidden, argmax_action, evalQ

    def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        return rnn_hidden, self.target_Qhead(q_inputs)

    def Q_tot(self, q, states=None):
        return self.eval_Qtot(q, states)

    def target_Q_tot(self, q, states=None):
        return self.target_Qtot(q, states)

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Qhead.trainable_params()

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
            tp.assign_value(ep)


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, **kwargs)
        self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
        self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
        self.q_feedforward = ff_mixer
        self.target_q_feedforward = copy.deepcopy(self.q_feedforward)
        self._concat = ms.ops.Concat(axis=-1)

    def q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
        else:
            outputs = self.representation(observation)
        q_inputs = self._concat([outputs['state'], agent_ids])
        return self.eval_Qhead_centralized(q_inputs)

    def target_q_centralized(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
        else:
            outputs = self.target_representation(observation)
        q_inputs = self._concat([outputs['state'], agent_ids])
        return self.target_Qhead_centralized(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead_centralized.trainable_params(), self.target_Qhead_centralized.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.q_feedforward.trainable_params(), self.target_q_feedforward.trainable_params()):
            tp.assign_value(ep)


class Qtran_MixingQnetwork(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = copy.deepcopy(qtran_mixer)
        self.q_tot = mixer
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = copy.deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return rnn_hidden, outputs['state'], argmax_action, evalQ

    def target_Q(self, observation: ms.Tensor, agent_ids: ms.Tensor, *rnn_hidden: ms.Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        return rnn_hidden, outputs['state'], self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.qtran_net.trainable_params(), self.target_qtran_net.trainable_params()):
            tp.assign_value(ep)


class DCG_policy(nn.Cell):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: Optional[Basic_Identical],
                 utility: Optional[nn.Cell] = None,
                 payoffs: Optional[nn.Cell] = None,
                 dcgraph: Optional[nn.Cell] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
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
                                   normalize, initialize, activation)
            self.target_bias = copy.deepcopy(self.bias)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor,
                  *rnn_hidden: ms.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs['state'], agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = copy.deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -9999999
            argmax_action = evalQ_detach.argmax(dim=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return rnn_hidden, argmax_action, evalQ

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.utility.trainable_params(), self.target_utility.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.payoffs.trainable_params(), self.target_payoffs.trainable_params()):
            tp.assign_value(ep)
        if self.dcg_s:
            for ep, tp in zip(self.bias.trainable_params(), self.target_bias.trainable_params()):
                tp.assign_value(ep)


class ActorNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class Basic_DDPG_policy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(Basic_DDPG_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        act = self.actor_net(actor_in)
        return outputs, act

    def critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        critic_in = self._concat([outputs['state'], actions, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        return self.target_actor_net(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class MADDPG_policy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(MADDPG_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
        self.parameters_critic = self.critic_net.trainable_params()
        self._concat = ms.ops.Concat(axis=-1)
        self._concat = ms.ops.Concat(axis=-1)
        self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))

    def construct(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        act = self.actor_net(actor_in)
        return outputs, act

    def critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.critic_net(critic_in)

    def target_critic(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        actions_n = self.broadcast_to(actions.view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions_n, agent_ids])
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: ms.Tensor, agent_ids: ms.Tensor):
        outputs = self.representation(observation)
        actor_in = self._concat([outputs['state'], agent_ids])
        return self.target_actor_net(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic_net.trainable_params(), self.target_critic_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))


class MATD3_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(MATD3_policy, self).__init__(action_space, n_agents, representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation)
        self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation)
        self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation)
        self.parameters_actor = self.representation.trainable_params() + self.actor_net.trainable_params()
        self.parameters_critic_A = self.critic_net_A.trainable_params()
        self.parameters_critic_B = self.critic_net_B.trainable_params()

        self.target_critic_net_A = copy.deepcopy(self.critic_net_A)
        self.target_critic_net_B = copy.deepcopy(self.critic_net_B)
        self.broadcast_to = ms.ops.BroadcastTo((-1, self.n_agents, -1))

    def Qpolicy(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions, agent_ids])
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, (qa + qb) / 2.0

    def Qtarget(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions, agent_ids])
        qa = self.target_critic_net_A(critic_in)
        qb = self.target_critic_net_B(critic_in)
        min_q = ms.ops.minimum(qa, qb)
        return outputs_n, min_q

    def Qaction_A(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions, agent_ids])
        qa = self.critic_net_A(critic_in)
        return outputs_n, qa

    def Qaction_B(self, observation: ms.Tensor, actions: ms.Tensor, agent_ids: ms.Tensor):
        bs = observation.shape[0]
        outputs_n = self.broadcast_to(self.representation(observation)['state'].view(bs, 1, -1))
        critic_in = self._concat([outputs_n, actions, agent_ids])
        qb = self.critic_net_B(critic_in)
        return outputs_n, qb

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.trainable_params(), self.target_actor_net.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic_net_A.trainable_params(), self.target_critic_net_A.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
        for ep, tp in zip(self.critic_net_B.trainable_params(), self.target_critic_net_B.trainable_params()):
            tp.assign_value((tau*ep.data+(1-tau)*tp.data))
