from operator import itemgetter
import mindspore as ms
from copy import deepcopy
from mindspore.nn.probability.distribution import Categorical
from gymnasium.spaces import Discrete, Box
from xuance.common import Sequence, Optional, Callable, Dict, List
from xuance.mindspore.utils import ModuleType
from xuance.mindspore import Tensor, Module, ModuleDict, ops
from .core import BasicQhead, ActorNet, CriticNet, VDN_mixer, QTRAN_base, QMIX_FF_mixer
from xuance.mindspore.representations import Basic_MLP


class BasicQnetwork(Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: ModuleType,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(BasicQnetwork, self).__init__()
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.representation_info_shape = {key: representation[key].output_shapes for key in self.model_keys}
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.representation = representation
        self.target_representation = deepcopy(self.representation)

        self.dim_input_Q, self.n_actions = {}, {}
        self.eval_Qhead, self.target_Qhead = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            self.dim_input_Q[key] = self.representation_info_shape[key]['state'][0]
            if self.use_parameter_sharing:
                self.dim_input_Q[key] += self.n_agents
            self.eval_Qhead[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                              normalize, initialize, activation)
            self.target_Qhead[key] = deepcopy(self.eval_Qhead[key])
            # update parameters name
            self.representation[key].update_parameters_name(key + '_rep_')
            self.eval_Qhead[key].update_parameters_name(key + '_eval_Qhead_')

        # MindSpore APIs
        self.argmax = ops.Argmax(output_type=ms.int32, axis=-1)

    @property
    def parameters_model(self):
        parameters_model = {}
        for key in self.model_keys:
            parameters_model[key] = self.representation[key].trainable_params() + \
                                    self.eval_Qhead[key].trainable_params()
        return parameters_model

    def construct(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                  avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                  rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, **kwargs):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            argmax_action (Dict[str, Tensor]): The actions output by the policies.
            evalQ (Dict[str, Tensor])： The evaluations of observation-action pairs.
        """
        rnn_hidden_new, argmax_action, evalQ = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                q_inputs = ops.cat([outputs, agent_ids], axis=-1)
            else:
                q_inputs = outputs

            evalQ[key] = self.eval_Qhead[key](q_inputs)

            if avail_actions is not None:
                evalQ_detach = ops.stop_gradient(evalQ[key].clone())
                evalQ_detach[avail_actions[key] == 0] = -1e10
                argmax_action[key] = self.argmax(evalQ_detach)
            else:
                argmax_action[key] = self.argmax(evalQ[key])

        return rnn_hidden_new, argmax_action, evalQ

    def Qtarget(self, observation: Dict[str, Tensor], agent_ids: Dict[str, Tensor],
                agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.target_representation[key](observation[key])
                rnn_hidden_new[key] = None
            if self.use_parameter_sharing:
                q_inputs = ops.cat([outputs, agent_ids], axis=-1)
            else:
                q_inputs = outputs
            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target

    def copy_target(self):
        for key in self.model_keys:
            for ep, tp in zip(self.representation[key].trainable_params(),
                              self.target_representation[key].trainable_params()):
                tp.assign_value(ep)
            for ep, tp in zip(self.eval_Qhead[key].trainable_params(),
                              self.target_Qhead[key].trainable_params()):
                tp.assign_value(ep)


class MixingQnetwork(BasicQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: ModuleDict,
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(MixingQnetwork, self).__init__(action_space, n_agents, representation, hidden_size,
                                             normalize, initialize, activation, **kwargs)
        self.eval_Qtot = mixer
        self.target_Qtot = deepcopy(self.eval_Qtot)

    def trainable_params(self, recurse=True):
        params = self.eval_Qtot.trainable_params() + self.representation.trainable_params() + self.eval_Qhead.trainable_params()
        return params

    def Q_tot(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        """
        Returns the total Q values.

        Parameters:
            individual_values (Dict[str, Tensor]): The individual Q values of all agents.
            states (Optional[Tensor]): The global states if necessary, default is None.

        Returns:
            evalQ_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = ops.cat([individual_values[k] for k in self.model_keys],
                                        axis=-1).reshape([-1, self.n_agents, 1])
        evalQ_tot = self.eval_Qtot(individual_inputs, states)
        return evalQ_tot

    def Qtarget_tot(self,
                    individual_values: Dict[str, Tensor],
                    states: Optional[Tensor] = None):
        """
        Returns the total Q values with target networks.

        Parameters:
            individual_values (Dict[str, Tensor]): The individual Q values of all agents.
            states (Optional[Tensor]): The global states if necessary, default is None. (Shape: batch * dim_state)

        Returns:
            q_target_tot (Tensor): The evaluated total Q values calculated by target networks.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = ops.cat([individual_values[k] for k in self.model_keys],
                                        axis=-1).reshape([-1, self.n_agents, 1])
        q_target_tot = self.target_Qtot(individual_inputs, states)
        return q_target_tot

    def copy_target(self):
        for key in self.model_keys:
            for ep, tp in zip(self.representation[key].trainable_params(),
                              self.target_representation[key].trainable_params()):
                tp.assign_value(ep)
            for ep, tp in zip(self.eval_Qhead[key].trainable_params(), self.target_Qhead[key].trainable_params()):
                tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
            tp.assign_value(ep)


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: Dict[str, Module],
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, **kwargs)
        self.eval_Qhead_centralized = deepcopy(self.eval_Qhead)
        self.target_Qhead_centralized = deepcopy(self.eval_Qhead_centralized)
        self.ff_mixer = ff_mixer
        self.target_ff_mixer = deepcopy(self.ff_mixer)
        # update parameters name for self.eval_Qhead_centralized
        for key in self.model_keys:
            self.eval_Qhead_centralized[key].update_parameters_name(key + '_eval_Qhead_centralized_')

    def trainable_params(self, recurse=True):
        params = self.eval_Qtot.trainable_params() + self.ff_mixer.trainable_params()
        for key in self.model_keys:
            params = (params + self.representation[key].trainable_params() + self.eval_Qhead[key].trainable_params() +
                      self.eval_Qhead_centralized[key].trainable_params())
        return params

    def q_centralized(self, observation: Dict[str, Tensor], agent_ids: Dict[str, Tensor],
                      agent_key: str = None, rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the centralised Q value.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            evalQ_cent (Tensor): The evaluated centralised Q values.
        """
        rnn_hidden_new, argmax_action, evalQ_cent = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                q_inputs = ops.cat([outputs, agent_ids], axis=-1)
            else:
                q_inputs = outputs

            evalQ_cent[key] = self.eval_Qhead_centralized[key](q_inputs)

        return rnn_hidden_new, evalQ_cent

    def target_q_centralized(self, observation: Dict[str, Tensor], agent_ids: Dict[str, Tensor],
                             agent_key: str = None, rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the centralised Q value with target networks.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target_cent (Tensor): The evaluated centralised Q values with target networks.
        """
        rnn_hidden_new, q_target_cent = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.target_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                q_inputs = ops.cat([outputs, agent_ids], axis=-1)
            else:
                q_inputs = outputs

            q_target_cent[key] = self.target_Qhead_centralized[key](q_inputs)

        return rnn_hidden_new, q_target_cent

    def q_feedforward(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        """
        Returns the total Q values with feedforward mixer networks.

        Parameters:
            individual_values (Dict[str, Tensor]): The individual Q values of all agents.
            states (Optional[Tensor]): The global states if necessary, default is None.

        Returns:
            evalQ_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = ops.cat([individual_values[k] for k in self.model_keys],
                                        axis=-1).reshape([-1, self.n_agents, 1])
        evalQ_tot = self.ff_mixer(individual_inputs, states)
        return evalQ_tot

    def target_q_feedforward(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        """
        Returns the total Q values with target feedforward mixer networks.

        Parameters:
            individual_values (Dict[str, Tensor]): The individual Q values of all agents.
            states (Optional[Tensor]): The global states if necessary, default is None.

        Returns:
            q_target_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = ops.cat([individual_values[k] for k in self.model_keys],
                                        axis=-1).reshape([-1, self.n_agents, 1])
        q_target_tot = self.target_ff_mixer(individual_inputs, states)
        return q_target_tot

    def copy_target(self):
        for key in self.model_keys:
            for ep, tp in zip(self.representation[key].trainable_params(),
                              self.target_representation[key].trainable_params()):
                tp.assign_value(ep)
            for ep, tp in zip(self.eval_Qhead[key].trainable_params(), self.target_Qhead[key].trainable_params()):
                tp.assign_value(ep)
            for ep, tp in zip(self.eval_Qhead_centralized[key].trainable_params(),
                              self.target_Qhead_centralized[key].trainable_params()):
                tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qtot.trainable_params(), self.target_Qtot.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.ff_mixer.trainable_params(), self.target_ff_mixer.trainable_params()):
            tp.assign_value(ep)


class Qtran_MixingQnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Dict[str, Module],
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation)
        self.target_Qhead = deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = deepcopy(qtran_mixer)
        self.q_tot = mixer
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: Tensor, agent_ids: Tensor,
                  *rnn_hidden: Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs, agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -1e10
            argmax_action = evalQ_detach.argmax(axis=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(axis=-1, keepdim=False)
        return rnn_hidden, outputs, argmax_action, evalQ

    def target_Q(self, observation: Tensor, agent_ids: Tensor, *rnn_hidden: Tensor):
        if self.use_rnn:
            outputs = self.target_representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.target_representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs, agent_ids])
        return rnn_hidden, outputs, self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.qtran_net.trainable_params(), self.target_qtran_net.trainable_params()):
            tp.assign_value(ep)


class DCG_policy(Module):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: Dict[str, Module],
                 utility: Optional[Module] = None,
                 payoffs: Optional[Module] = None,
                 dcgraph: Optional[Module] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(DCG_policy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(self.representation)
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.utility = utility
        self.target_utility = deepcopy(self.utility)
        self.payoffs = payoffs
        self.target_payoffs = deepcopy(self.payoffs)
        self.graph = dcgraph
        self.dcg_s = False
        if hidden_size_bias is not None:
            self.dcg_s = True
            self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                   normalize, initialize, activation)
            self.target_bias = deepcopy(self.bias)
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, observation: Tensor, agent_ids: Tensor,
                  *rnn_hidden: Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        q_inputs = self._concat([outputs, agent_ids])
        evalQ = self.eval_Qhead(q_inputs)
        if avail_actions is not None:
            evalQ_detach = deepcopy(evalQ)
            evalQ_detach[avail_actions == 0] = -1e10
            argmax_action = evalQ_detach.argmax(axis=-1, keepdim=False)
        else:
            argmax_action = evalQ.argmax(axis=-1, keepdim=False)
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


class MFQnetwork(Module):
    """
    The base class to implement Mean Field Reinforcement Learning - MFQ.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The action space, which type is gym.spaces.Discrete.
        n_agents (int): The number of agents.
        representation (ModuleDict): A dict of the representation module for all agents.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters' initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Dict[str, Module],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MFQnetwork, self).__init__()
        self.action_space = action_space
        self.n_agents = n_agents
        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = int(max(self.n_actions_list))
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.representation_info_shape = {key: representation[key].output_shapes for key in self.model_keys}
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        # The choice of policy: Boltzmann policy or greedy policy. (Default is 'greedy')
        self.policy_type = kwargs['policy_type']

        self.representation = representation
        self.target_representation = deepcopy(self.representation)

        self.dim_input_action_embedding, self.dim_input_Q, self.n_actions = {}, {}, {}
        self.action_mean_embedding = ModuleDict()
        self.eval_Qhead, self.target_Qhead, self.target_action_mean_embedding = ModuleDict(), ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.dim_input_action_embedding[key] = self.n_actions_max
            self.dim_input_Q[key] = self.representation_info_shape[key]['state'][0] + \
                                    kwargs['action_embedding_hidden_size'][-1]
            self.n_actions[key] = self.action_space[key].n
            if self.use_parameter_sharing:
                self.dim_input_action_embedding[key] += self.n_agents
                self.dim_input_Q[key] += self.n_agents
            self.action_mean_embedding[key] = Basic_MLP((self.dim_input_action_embedding[key],),
                                                        kwargs['action_embedding_hidden_size'],
                                                        normalize, initialize, activation)
            self.eval_Qhead[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                              normalize, initialize, activation)
            self.target_action_mean_embedding[key] = deepcopy(self.action_mean_embedding[key])
            self.target_Qhead[key] = deepcopy(self.eval_Qhead[key])
            # update parameters name
            self.representation[key].update_parameters_name(key + '_rep_')
            self.action_mean_embedding[key].update_parameters_name(key + '_act_embedding_')
            self.eval_Qhead[key].update_parameters_name(key + '_eval_Qhead_')
        self.softmax = ops.Softmax(axis=-1)
        self.temperature = kwargs['temperature']

    @property
    def parameters_model(self):
        parameters_model = {}
        for key in self.model_keys:
            parameters_model[key] = self.representation[key].trainable_params() + self.action_mean_embedding[
                key].trainable_params() + self.eval_Qhead[key].trainable_params()
        return parameters_model

    def construct(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                actions_mean: Dict[str, Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            actions_mean (Dict[str, Tensor]): The mean actions of each agent's neighbors.
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            argmax_action (Dict[str, Tensor]): The actions output by the policies.
            evalQ (Dict[str, Tensor])： The evaluations of observation-action pairs.
        """
        rnn_hidden_new, actions, evalQ = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        actions_mean = {key: Tensor(actions_mean[key]) for key in agent_list}
        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            # mean actions embedding
            if self.use_parameter_sharing:
                action_embedding_input = ops.cat([actions_mean[key], agent_ids], axis=-1)
                act_embedding = self.action_mean_embedding[key](action_embedding_input)
                q_inputs = ops.cat([outputs, act_embedding, agent_ids], axis=-1)
            else:
                act_embedding = self.action_mean_embedding[key](actions_mean[key])
                q_inputs = ops.cat([outputs, act_embedding], axis=-1)

            evalQ[key] = self.eval_Qhead[key](q_inputs)

            evalQ_detach = ops.stop_gradient(deepcopy(evalQ[key]))
            if avail_actions is not None:
                evalQ_detach[avail_actions[key] == 0] = -1e10

            if self.policy_type == "Boltzmann":
                actions_prob = self.get_boltzmann_policy(evalQ_detach)
                actions[key] = Categorical(probs=actions_prob).sample()
            elif self.policy_type == "greedy":
                actions[key] = evalQ_detach.argmax(axis=-1, keepdim=False)
            else:
                raise NotImplementedError

        return rnn_hidden_new, actions, evalQ

    def get_boltzmann_policy(self, q):
        actions_prob = self.softmax(q / self.temperature)
        return actions_prob

    def get_mean_actions(self, actions: Dict[str, Tensor],
                         agent_mask_tensor: Tensor, batch_size: int):
        if self.use_parameter_sharing:
            actions_tensor = actions[self.model_keys[0]].reshape([-1, self.n_agents])
        else:
            actions_tensor = ops.stack(itemgetter(*self.model_keys)(actions), axis=-1).reshape([-1, self.n_agents])
        actions_onehot = ops.one_hot(actions_tensor, depth=self.n_actions_max)

        # count alive neighbors
        _eyes = ops.repeat_elements(ops.eye(self.n_agents).unsqueeze(0), rep=batch_size, axis=0)
        agent_mask_diagonal = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1), rep=self.n_agents, axis=2) * _eyes
        agent_mask_neighbors = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1),
                                                   rep=self.n_agents, axis=2) - agent_mask_diagonal
        agent_alive_neighbors = agent_mask_neighbors.sum(axis=-1, keepdims=True)

        # calculate mean actions of each agent's neighbors
        agent_mask_repeat = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1), rep=self.n_actions_max, axis=2)
        actions_onehot = actions_onehot * agent_mask_repeat
        actions_sum = ops.repeat_elements(actions_onehot.sum(axis=-2, keepdims=True), rep=self.n_agents, axis=1)
        actions_neighbors_sum = actions_sum - actions_onehot  # Sum of other agents' actions.
        actions_mean_masked = actions_neighbors_sum * agent_mask_repeat / agent_alive_neighbors
        return actions_mean_masked

    def Qtarget(self, observation: Dict[str, Tensor], actions_mean: Dict[str, Tensor],
                agent_ids: Dict[str, Tensor],
                agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            actions_mean (Dict[str, Tensor]): The mean of each agent's neighbors.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        actions_mean = {key: Tensor(actions_mean[key]) for key in agent_list}
        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.target_representation[key](observation[key])
                rnn_hidden_new[key] = None

            # mean actions embedding
            if self.use_parameter_sharing:
                input_embedding = ops.cat([actions_mean[key], agent_ids], axis=-1)
                act_embedding = self.target_action_mean_embedding[key](input_embedding)
                q_inputs = ops.cat([outputs, act_embedding, agent_ids], axis=-1)
            else:
                act_embedding = self.target_action_mean_embedding[key](actions_mean[key])
                q_inputs = ops.cat([outputs, act_embedding], axis=-1)

            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.action_mean_embedding.trainable_params(), self.target_action_mean_embedding.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Qhead.trainable_params(), self.target_Qhead.trainable_params()):
            tp.assign_value(ep)


class Independent_DDPG_Policy(Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Dict[str, Module],
                 critic_representation: Dict[str, Module],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 **kwargs):
        super(Independent_DDPG_Policy, self).__init__()
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.actor_representation_info_shape = {key: actor_representation[key].output_shapes for key in self.model_keys}
        self.critic_representation_info_shape = {key: critic_representation[key].output_shapes
                                                 for key in self.model_keys}
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = actor_representation
        self.critic_representation = critic_representation
        self.target_actor_representation = deepcopy(self.actor_representation)
        self.target_critic_representation = deepcopy(self.critic_representation)

        self.actor, self.target_actor = ModuleDict(), ModuleDict()
        self.critic, self.target_critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                       normalize, initialize, activation, activation_action)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_actor[key] = deepcopy(self.actor[key])
            self.target_critic[key] = deepcopy(self.critic[key])
            # update parameters name
            self.actor_representation[key].update_parameters_name(key + '_rep_actor_')
            self.critic_representation[key].update_parameters_name(key + '_rep_critic_')
            self.actor[key].update_parameters_name(key + '_actor_')
            self.critic[key].update_parameters_name(key + '_critic_')

    @property
    def parameters_actor(self):
        parameters_actor = {}
        for key in self.model_keys:
            parameters_actor[key] = self.actor_representation[key].trainable_params() + self.actor[
                key].trainable_params()
        return parameters_actor

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = self.critic_representation[key].trainable_params() + self.critic[
                key].trainable_params()
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions.
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep + dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def construct(self, observation: Dict[str, Tensor],
                  agent_ids: Tensor = None, agent_key: str = None,
                  rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            actions (Dict[Tensor]): The actions output by the policies.
        """
        rnn_hidden_new, actions = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.actor_representation[key](observation[key])

            if self.use_parameter_sharing:
                actor_in = ops.cat([outputs, agent_ids], axis=-1)
            else:
                actor_in = outputs
            actions[key] = self.actor[key](actor_in)
        return rnn_hidden_new, actions

    def Qpolicy(self, observation: Dict[str, Tensor], actions: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            actions (Dict[Tensor]): The actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_eval: The evaluations of Q^policy.
        """
        rnn_hidden_new, q_eval = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.critic_representation[key](observation[key])

            if self.use_parameter_sharing:
                critic_in = ops.cat([outputs, agent_ids], axis=-1)
            else:
                critic_in = outputs
            q_eval[key] = self.critic[key](ops.cat([critic_in, actions[key]], axis=-1))
        return rnn_hidden_new, q_eval

    def Qtarget(self, next_observation: Dict[str, Tensor], next_actions: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            next_actions (Dict[Tensor]): The actions of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_new, q_target = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_critic_representation[key](next_observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.target_critic_representation[key](next_observation[key])

            if self.use_parameter_sharing:
                critic_in = ops.cat([outputs, agent_ids], axis=-1)
            else:
                critic_in = outputs
            q_target[key] = self.target_critic[key](ops.cat([critic_in, next_actions[key]], axis=-1))
        return rnn_hidden_new, q_target

    def Atarget(self, next_observation: Dict[str, Tensor],
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the next actions by target policies.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            next_actions (Dict[Tensor]): The next actions.
        """
        rnn_hidden_new, next_actions = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.target_actor_representation[key](next_observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.target_actor_representation[key](next_observation[key])

            if self.use_parameter_sharing:
                actor_in = ops.cat([outputs, agent_ids], axis=-1)
            else:
                actor_in = outputs
            next_actions[key] = self.target_actor[key](actor_in)
        return rnn_hidden_new, next_actions

    def soft_update(self, tau=0.005):
        for key in self.model_keys:
            for ep, tp in zip(self.actor_representation[key].trainable_params(),
                              self.target_actor_representation[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic_representation[key].trainable_params(),
                              self.target_critic_representation[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.actor[key].trainable_params(), self.target_actor[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic[key].trainable_params(), self.target_critic[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))


class MADDPG_Policy(Independent_DDPG_Policy):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Dict[str, Module],
                 critic_representation: Dict[str, Module],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 **kwargs):
        super(MADDPG_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, activation_action, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_action: The dimension of actions.
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self, joint_observation: Tensor, joint_actions: Tensor,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (Tensor): The joint observations of the team.
            joint_actions (Tensor): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_eval: The evaluations of Q^policy.
        """
        rnn_hidden_new, q_eval = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = ops.cat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs = {k: self.critic_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            rnn_hidden_new.update({k: (outputs[k]['rnn_hidden'], outputs[k]['rnn_cell']) for k in agent_list})
        else:
            outputs = {k: self.critic_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out = outputs[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out = joint_rep_out.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out = outputs[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out = joint_rep_out.reshape(bs, -1)
                critic_in = ops.cat([joint_rep_out, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out = outputs[key].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out = outputs[key].reshape(bs, -1)
                critic_in = joint_rep_out
            q_eval[key] = self.critic[key](critic_in)
        return rnn_hidden_new, q_eval

    def Qtarget(self, joint_observation: Tensor, joint_actions: Tensor,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (Tensor): The joint observations of the team.
            joint_actions (Tensor): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_new, q_target = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = ops.cat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs = {k: self.target_critic_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            rnn_hidden_new.update({k: (outputs[k]['rnn_hidden'], outputs[k]['rnn_cell']) for k in agent_list})
        else:
            outputs = {k: self.target_critic_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out = outputs[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out = joint_rep_out.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out = outputs[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out = joint_rep_out.reshape(bs, -1)
                critic_in = ops.cat([joint_rep_out, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out = outputs[key].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out = outputs[key].reshape(bs, -1)
                critic_in = joint_rep_out
            q_target[key] = self.target_critic[key](critic_in)
        return rnn_hidden_new, q_target


class MATD3_Policy(MADDPG_Policy, Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Dict[str, Module],
                 critic_representation: Dict[str, Module],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 **kwargs):
        Module.__init__(self)
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.actor_representation_info_shape = {key: actor_representation[key].output_shapes for key in self.model_keys}
        self.critic_representation_info_shape = {key: critic_representation[key].output_shapes for key in
                                                 self.model_keys}
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = actor_representation
        self.critic_A_representation = critic_representation
        self.critic_B_representation = deepcopy(critic_representation)
        self.target_actor_representation = deepcopy(self.actor_representation)
        self.target_critic_A_representation = deepcopy(self.critic_A_representation)
        self.target_critic_B_representation = deepcopy(self.critic_B_representation)

        self.actor, self.target_actor = ModuleDict(), ModuleDict()
        self.critic_A, self.critic_B = ModuleDict(), ModuleDict()
        self.target_critic_A, self.target_critic_B = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_A_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                       normalize, initialize, activation, activation_action)
            self.critic_A[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.critic_B[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_actor[key] = deepcopy(self.actor[key])
            self.target_critic_A[key] = deepcopy(self.critic_A[key])
            self.target_critic_B[key] = deepcopy(self.critic_B[key])
            # Update parameters name
            self.actor_representation[key].update_parameters_name(key + '_rep_actor_')
            self.critic_A_representation[key].update_parameters_name(key + '_rep_critic_A_')
            self.critic_B_representation[key].update_parameters_name(key + '_rep_critic_B_')
            self.actor[key].update_parameters_name(key + '_actor_')
            self.critic_A[key].update_parameters_name(key + '_critic_A_')
            self.critic_B[key].update_parameters_name(key + '_critic_B_')

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = self.critic_A_representation[key].trainable_params() + \
                                     self.critic_A[key].trainable_params() + \
                                     self.critic_B_representation[key].trainable_params() + \
                                     self.critic_B[key].trainable_params()
        return parameters_critic

    def Qpolicy(self, joint_observation: Tensor, joint_actions: Tensor,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (Tensor): The joint observations of the team.
            joint_actions (Tensor): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            q_eval_A (Dict[Tensor]): The evaluations of Q^policy calculated by critic A.
            q_eval_B (Dict[Tensor]): The evaluations of Q^policy calculated by critic B.
            q_eval (Dict[Tensor]): The evaluations of Q^policy averaged by critic A and Critic B.
        """
        q_eval, q_eval_A, q_eval_B = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = ops.cat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_A = {k: self.critic_A_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            outputs_B = {k: self.critic_B_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
        else:
            outputs_A = {k: self.critic_A_representation[k](critic_rep_in) for k in agent_list}
            outputs_B = {k: self.critic_B_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_A = outputs_A[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out_B = outputs_B[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out_A = joint_rep_out_A.reshape(bs, seq_len, -1)
                    joint_rep_out_B = joint_rep_out_B.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_A = outputs_A[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out_B = outputs_B[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out_A = joint_rep_out_A.reshape(bs, -1)
                    joint_rep_out_B = joint_rep_out_B.reshape(bs, -1)
                critic_in_A = ops.cat([joint_rep_out_A, agent_ids], axis=-1)
                critic_in_B = ops.cat([joint_rep_out_B, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_A = outputs_A[key].reshape(bs, seq_len, -1)
                    joint_rep_out_B = outputs_B[key].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_A = outputs_A[key].reshape(bs, -1)
                    joint_rep_out_B = outputs_B[key].reshape(bs, -1)
                critic_in_A = joint_rep_out_A
                critic_in_B = joint_rep_out_B
            q_eval_A[key] = self.critic_A[key](critic_in_A)
            q_eval_B[key] = self.critic_B[key](critic_in_B)
            q_eval[key] = (q_eval_A[key] + q_eval_B[key]) / 2.0

        return q_eval_A, q_eval_B, q_eval

    def Qtarget(self, joint_observation: Tensor, joint_actions: Tensor,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (Tensor): The joint observations of the team.
            joint_actions (Tensor): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            q_target (Dict[Tensor]): The evaluations of Q^target.
        """
        q_target = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = ops.cat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_A = {k: self.target_critic_A_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            outputs_B = {k: self.target_critic_B_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
        else:
            outputs_A = {k: self.target_critic_A_representation[k](critic_rep_in) for k in agent_list}
            outputs_B = {k: self.target_critic_B_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_A = outputs_A[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out_B = outputs_B[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1, -1))
                    joint_rep_out_A = joint_rep_out_A.reshape(bs, seq_len, -1)
                    joint_rep_out_B = joint_rep_out_B.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_A = outputs_A[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out_B = outputs_B[key].unsqueeze(1).broadcast_to((-1, self.n_agents, -1))
                    joint_rep_out_A = joint_rep_out_A.reshape(bs, -1)
                    joint_rep_out_B = joint_rep_out_B.reshape(bs, -1)
                critic_in_A = ops.cat([joint_rep_out_A, agent_ids], axis=-1)
                critic_in_B = ops.cat([joint_rep_out_B, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_A = outputs_A[key].reshape(bs, seq_len, -1)
                    joint_rep_out_B = outputs_B[key].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_A = outputs_A[key].reshape(bs, -1)
                    joint_rep_out_B = outputs_B[key].reshape(bs, -1)
                critic_in_A = joint_rep_out_A
                critic_in_B = joint_rep_out_B
            q_target_A = self.target_critic_A[key](critic_in_A)
            q_target_B = self.target_critic_B[key](critic_in_B)
            q_target[key] = ops.minimum(q_target_A, q_target_B)

        return q_target

    def soft_update(self, tau=0.005):
        for key in self.model_keys:
            for ep, tp in zip(self.actor_representation[key].trainable_params(),
                              self.target_actor_representation[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic_A_representation[key].trainable_params(),
                              self.target_critic_A_representation[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic_B_representation[key].trainable_params(),
                              self.target_critic_B_representation[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.actor[key].trainable_params(), self.target_actor[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic_A[key].trainable_params(), self.target_critic_A[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
            for ep, tp in zip(self.critic_B[key].trainable_params(), self.target_critic_B[key].trainable_params()):
                tp.assign_value((tau * ep.data + (1 - tau) * tp.data))
