import numpy as np
from copy import deepcopy
from gym.spaces import Space, Discrete
from xuance.common import Sequence, Optional, Union, Dict, List
from xuance.tensorflow.representations import Basic_Identical
from xuance.tensorflow import tf, tk, tfp, Tensor, Module
from .core import BasicQhead, ActorNet, CriticNet, VDN_mixer, QTRAN_base


class BasicQnetwork(Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: Union[Basic_Identical, dict],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
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
        self.eval_Qhead, self.target_Qhead = {}, {}
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            self.dim_input_Q[key] = self.representation_info_shape[key]['state'][0]
            if self.use_parameter_sharing:
                self.dim_input_Q[key] += self.n_agents
            self.eval_Qhead[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                              normalize, initialize, activation)
            self.target_Qhead[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                                normalize, initialize, activation)
            self.target_Qhead[key].set_weights(self.eval_Qhead[key].get_weights())

    @property
    def parameters_model(self):
        parameters_model = {}
        for key in self.model_keys:
            parameters_model[key] = self.representation[key].trainable_variables + self.eval_Qhead[
                key].trainable_variables
        return parameters_model

    @tf.function
    def call(self, observation: Dict[str, np.ndarray], agent_ids: np.ndarray = None,
             avail_actions: Dict[str, np.ndarray] = None, agent_key: str = None,
             rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None, **kwargs):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, np.ndarray]): The input observations for the policies.
            agent_ids (np.ndarray): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, np.ndarray]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[np.ndarray]]]): The new hidden variables of the RNN.
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
                q_inputs = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                q_inputs = outputs['state']

            evalQ[key] = self.eval_Qhead[key](q_inputs)

            if avail_actions is not None:
                evalQ_detach = tf.stop_gradient(evalQ[key].clone())
                evalQ_detach[avail_actions[key] == 0] = -9999999
                argmax_action[key] = tf.argmax(evalQ_detach, axis=-1)
            else:
                argmax_action[key] = tf.argmax(evalQ[key], axis=-1)

        return rnn_hidden_new, argmax_action, evalQ

    @tf.function
    def Qtarget(self, observation: Dict[str, np.ndarray], agent_ids: Dict[str, np.ndarray],
                agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            observation (Dict[np.ndarray]): The observations.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[np.ndarray]]]): The new hidden variables of the RNN.
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
                q_inputs = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                q_inputs = outputs['state']
            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target

    def copy_target(self):
        for key in self.model_keys:
            self.target_representation[key].set_weights(self.representation[key].get_weights())
            self.target_Qhead[key].set_weights(self.eval_Qhead[key].get_weights())


class MixingQnetwork(BasicQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: Union[Module, dict],
                 mixer: Optional[List[Module]] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 **kwargs):
        super(MixingQnetwork, self).__init__(action_space, n_agents, representation, hidden_size,
                                             normalize, initialize, activation, **kwargs)
        self.eval_Qtot = mixer[0]
        self.target_Qtot = mixer[1]
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())

    @tf.function
    def Q_tot(self, individual_values: Dict[str, np.ndarray], states: Optional[np.ndarray] = None):
        """
        Returns the total Q values.

        Parameters:
            individual_values (Dict[str, np.ndarray]): The individual Q values of all agents.
            states (Optional[np.ndarray]): The global states if necessary, default is None.

        Returns:
            evalQ_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = tf.reshape(individual_values[self.model_keys[0]], [-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = tf.reshape(tf.concat([individual_values[k] for k in self.model_keys],
                                                     axis=-1), [-1, self.n_agents, 1])
        evalQ_tot = self.eval_Qtot(individual_inputs, states)
        return evalQ_tot

    @tf.function
    def Qtarget_tot(self,
                    individual_values: Dict[str, np.ndarray],
                    states: Optional[np.ndarray] = None):
        """
        Returns the total Q values with target networks.

        Parameters:
            individual_values (Dict[str, np.ndarray]): The individual Q values of all agents.
            states (Optional[np.ndarray]): The global states if necessary, default is None. (Shape: batch * dim_state)

        Returns:
            q_target_tot (Tensor): The evaluated total Q values calculated by target networks.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = tf.reshape(individual_values[self.model_keys[0]], [-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = tf.reshape(tf.concat([individual_values[k] for k in self.model_keys],
                                                     axis=-1), [-1, self.n_agents, 1])
        q_target_tot = self.target_Qtot(individual_inputs, states)
        return q_target_tot

    def copy_target(self):
        for key in self.model_keys:
            self.target_representation[key].set_weights(self.representation.get_weights())
            self.target_Qhead[key].set_weights(self.eval_Qhead[key].get_weights())
        self.target_Qtot.set_weights(self.eval_Qtot.get_weights())


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[List[Module]] = None,
                 ff_mixer: Optional[List[Module]] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 **kwargs):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, **kwargs)
        self.eval_Qhead_centralized, self.target_Qhead_centralized = {}, {}
        for key in self.model_keys:
            self.eval_Qhead_centralized[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                                          normalize, initialize, activation)
            self.target_Qhead_centralized[key] = BasicQhead(self.dim_input_Q[key], self.n_actions[key], hidden_size,
                                                            normalize, initialize, activation)
            self.target_Qhead_centralized[key].set_weights(self.eval_Qhead_centralized[key].get_weights())

        self.ff_mixer = ff_mixer[0]
        self.target_ff_mixer = ff_mixer[1]
        self.target_ff_mixer.set_weights(self.ff_mixer.get_weights())

    @tf.function
    def q_centralized(self, observation: Dict[str, np.ndarray], agent_ids: Dict[str, np.ndarray],
                      agent_key: str = None, rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the centralised Q value.

        Parameters:
            observation (Dict[np.ndarray]): The observations.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                q_inputs = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                q_inputs = outputs['state']

            evalQ_cent[key] = self.eval_Qhead_centralized[key](q_inputs)

        return rnn_hidden_new, evalQ_cent

    @tf.function
    def target_q_centralized(self, observation: Dict[str, np.ndarray], agent_ids: Dict[str, np.ndarray],
                             agent_key: str = None, rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the centralised Q value with target networks.

        Parameters:
            observation (Dict[np.ndarray]): The observations.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                q_inputs = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                q_inputs = outputs['state']

            q_target_cent[key] = self.target_Qhead_centralized[key](q_inputs)

        return rnn_hidden_new, q_target_cent

    @tf.function
    def q_feedforward(self, individual_values: Union[Dict[str, np.ndarray], Dict[str, Tensor]],
                      states: Optional[np.ndarray] = None):
        """
        Returns the total Q values with feedforward mixer networks.

        Parameters:
            individual_values (Union[Dict[str, np.ndarray], Dict[str, Tensor]]): The individual Q values of all agents.
            states (Optional[np.ndarray]): The global states if necessary, default is None.

        Returns:
            evalQ_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = tf.reshape(individual_values[self.model_keys[0]], [-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = tf.concat([individual_values[k] for k in self.model_keys],
                                          axis=-1).reshape([-1, self.n_agents, 1])
        evalQ_tot = self.ff_mixer(individual_inputs, states)
        return evalQ_tot

    @tf.function
    def target_q_feedforward(self, individual_values: Dict[str, np.ndarray],
                             states: Optional[np.ndarray] = None):
        """
        Returns the total Q values with target feedforward mixer networks.

        Parameters:
            individual_values (Dict[str, np.ndarray]): The individual Q values of all agents.
            states (Optional[np.ndarray]): The global states if necessary, default is None.

        Returns:
            q_target_tot (Tensor): The evaluated total Q values for the multi-agent team.
        """
        if self.use_parameter_sharing:
            """
            From dict to tensor. For example:
                individual_values: {'agent_0': batch * n_agents * 1} -> 
                individual_inputs: batch * n_agents * 1
            """
            individual_inputs = tf.reshape(individual_values[self.model_keys[0]], [-1, self.n_agents, 1])
        else:
            """
            From dict to tensor. For example: 
                individual_values: {'agent_0': batch * 1, 'agent_1': batch * 1, 'agent_2': batch * 1} -> 
                individual_inputs: batch * 2 * 1
            """
            individual_inputs = tf.reshape(tf.concat([individual_values[k] for k in self.model_keys],
                                                     axis=-1), [-1, self.n_agents, 1])
        q_target_tot = self.target_ff_mixer(individual_inputs, states)
        return q_target_tot

    def copy_target(self):
        for key in self.model_keys:
            self.target_representation[key].set_weights(self.representation[key].get_weights())
            self.target_Qhead[key].set_weights(self.eval_Qhead[key].get_weights())
            self.target_Qhead_centralized[key].set_weights(self.eval_Qhead_centralized[key].get_weights())
        self.target_ff_mixer.set_weights(self.ff_mixer.get_weights())


class Qtran_MixingQnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.obs_dim = self.representation.input_shapes[0]
        self.hidden_state_dim = self.representation.output_shapes['state'][0]
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                       hidden_size, normalize, initializer, activation, device)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = qtran_mixer
        self.q_tot = mixer
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_qtran_net.set_weights(self.qtran_net.get_weights())

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = tf.reshape(self.eval_Qhead(q_inputs), [-1, self.n_agents, self.action_dim])
        argmax_action = tf.argmax(evalQ, axis=-1)
        return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), argmax_action, evalQ

    def target_Q(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.target_representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        return tf.reshape(outputs['state'], [-1, self.n_agents, self.hidden_state_dim]), self.target_Qhead(q_inputs)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())
        self.target_qtran_net.set_weights(self.qtran_net.get_weights())


class DCG_policy(Module):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: Optional[Basic_Identical],
                 utility: Optional[Module] = None,
                 payoffs: Optional[Module] = None,
                 dcgraph: Optional[Module] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0",
                 **kwargs):
        super(DCG_policy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = representation
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.utility = utility
        self.target_utility = utility
        self.payoffs = payoffs
        self.target_payoffs = payoffs
        self.graph = dcgraph
        self.dcg_s = False
        if hidden_size_bias is not None:
            self.dcg_s = True
            self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                   normalize, initializer, activation, device)
            self.target_bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                          normalize, initializer, activation, device)

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden: Tensor, **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        q_inputs = tf.concat([outputs['state'], IDs], axis=-1)
        evalQ = self.eval_Qhead(q_inputs)
        evalQ = tf.reshape(evalQ, [-1, self.n_agents, self.action_dim])
        argmax_action = tf.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_utility.set_weights(self.utility.get_weights())
        self.target_payoffs.set_weights(self.payoffs.get_weights())
        if self.dcg_s:
            self.target_bias.set_weights(self.bias.get_weights())


class MFQnetwork(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
        super(MFQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, hidden_size, normalize, initializer, activation, device)
        self.target_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                       n_agents, hidden_size, normalize, initializer, activation, device)
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observation = inputs["obs"]
        actions_mean = inputs["act_mean"]
        agent_ids = inputs["ids"]
        outputs = self.representation(observation)
        q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = tf.argmax(evalQ, axis=-1)
        return outputs, argmax_action, evalQ

    def sample_actions(self, logits: Tensor):
        dist = tfp.distributions.Categorical(logits=logits)
        return dist.sample()

    def target_Q(self, observation: Tensor, actions_mean: Tensor, agent_ids: Tensor):
        outputs = self.target_representation(observation)
        q_inputs = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        self.target_representation.set_weights(self.representation.get_weights())
        self.target_Qhead.set_weights(self.eval_Qhead.get_weights())


class Independent_DDPG_Policy(Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(Independent_DDPG_Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.obs_dim = self.representation.input_shapes[0]
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initializer, activation, device)
        if isinstance(self.representation, Basic_Identical):
            self.parameters_actor = self.actor_net.trainable_variables
        else:
            self.parameters_actor = self.representation.trainable_variables + self.actor_net.trainable_variables
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(1.0)

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.actor_net(actor_in)
        return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        observation = tf.reshape(observation, [-1, self.obs_dim])
        actions = tf.reshape(actions, [-1, self.action_dim])
        agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return tf.reshape(self.critic_net(critic_in), [-1, self.n_agents, 1])

    def target_critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        observation = tf.reshape(observation, [-1, self.obs_dim])
        actions = tf.reshape(actions, [-1, self.action_dim])
        agent_ids = tf.reshape(agent_ids, [-1, self.n_agents])
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions, agent_ids], axis=-1)
        return tf.reshape(self.target_critic_net(critic_in), [-1, self.n_agents, 1])

    def target_actor(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.target_actor_net(actor_in)
        return tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net.variables, self.target_critic_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)


class MADDPG_Policy(Independent_DDPG_Policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(MADDPG_Policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initializer, activation, device)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                           critic_hidden_size, normalize, initializer, activation, device)
        self.parameters_critic = self.critic_net.trainable_variables
        self.soft_update(1.0)

    def critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        return self.target_critic_net(critic_in)


class MATD3_Policy(Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"
                 ):
        super(MATD3_Policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.obs_dim = self.representation.input_shapes[0]
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation, device)
        self.target_actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                         actor_hidden_size, normalize, initializer, activation, device)
        self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initializer, activation, device)
        self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                             critic_hidden_size, normalize, initializer, activation, device)
        self.target_critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                             critic_hidden_size, normalize, initializer, activation, device)
        self.soft_update(tau=1.0)
        self.critic_parameters = self.critic_net_A.trainable_variables + self.critic_net_B.trainable_variables

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.actor_net(actor_in)
        return outputs, tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, (qa + qb) / 2.0

    def target_critic(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.target_critic_net_A(critic_in)
        qb = self.target_critic_net_B(critic_in)
        min_q = tf.math.minimum(qa, qb)
        return outputs_n, min_q

    def Qaction(self, observation: Tensor, actions: Tensor, agent_ids: Tensor):
        bs = observation.shape[0]
        outputs_n = tf.reshape(self.representation(observation)['state'], (bs, 1, -1))
        outputs_n = tf.tile(outputs_n, (1, self.n_agents, 1))
        actions_n = tf.tile(tf.reshape(actions, (bs, 1, -1)), (1, self.n_agents, 1))
        critic_in = tf.concat([outputs_n, actions_n, agent_ids], axis=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, tf.concat((qa, qb), axis=-1)

    def target_actor(self, inputs: Union[np.ndarray, dict]):
        observations = tf.reshape(inputs['obs'], [-1, self.obs_dim])
        IDs = tf.reshape(inputs['ids'], [-1, self.n_agents])
        outputs = self.representation(observations)
        actor_in = tf.concat([outputs['state'], IDs], axis=-1)
        act = self.target_actor_net(actor_in)
        return tf.reshape(act, [-1, self.n_agents, self.action_dim])

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.variables, self.target_actor_net.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net_A.variables, self.target_critic_net_A.variables):
            tp.assign((1 - tau) * tp + tau * ep)
        for ep, tp in zip(self.critic_net_B.variables, self.target_critic_net_B.variables):
            tp.assign((1 - tau) * tp + tau * ep)
