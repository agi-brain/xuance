import numpy as np
from copy import deepcopy
from gym.spaces import Space, Discrete, Box
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
            parameters_model[key] = self.representation[key].trainable_variables + \
                                    self.eval_Qhead[key].trainable_variables
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
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Optional[Dict[str, Module]],
                 critic_representation: Optional[Dict[str, Module]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
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

        self.actor, self.target_actor, self.critic, self.target_critic = {}, {}, {}, {}
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                       normalize, initialize, activation, activation_action)
            self.target_actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                              normalize, initialize, activation, activation_action)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_actor[key].set_weights(self.actor[key].get_weights())
            self.target_critic[key].set_weights(self.critic[key].get_weights())

    def actor_trainable_variables(self, key):
        return self.actor_representation[key].trainable_variables + self.actor[key].trainable_variables

    def critic_trainable_variables(self, key):
        return self.critic_representation[key].trainable_variables + self.critic[key].trainable_variables

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

    @tf.function
    def call(self, observation: Dict[str, np.ndarray],
             agent_ids: np.ndarray = None, agent_key: str = None,
             rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[np.ndarray]): The input observations for the policies.
            agent_ids (np.ndarray): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_in = outputs['state']
            actions[key] = self.actor[key](actor_in)
        return rnn_hidden_new, actions

    @tf.function
    def Qpolicy(self, observation: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[np.ndarray]): The observations.
            actions (Dict[np.ndarray]): The actions.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                critic_in = outputs['state']
            q_eval[key] = self.critic[key](tf.concat([critic_in, actions[key]], axis=-1))
        return rnn_hidden_new, q_eval

    @tf.function
    def Qtarget(self, next_observation: Dict[str, np.ndarray], next_actions: Dict[str, np.ndarray],
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[np.ndarray]): The observations of next step.
            next_actions (Dict[np.ndarray]): The actions of next step.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                critic_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                critic_in = outputs['state']
            q_target[key] = self.target_critic[key](tf.concat([critic_in, next_actions[key]], axis=-1))
        return rnn_hidden_new, q_target

    @tf.function
    def Atarget(self, next_observation: Dict[str, np.ndarray],
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the next actions by target policies.

        Parameters:
            next_observation (Dict[np.ndarray]): The observations of next step.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

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
                actor_in = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_in = outputs['state']
            next_actions[key] = self.target_actor[key](actor_in)
        return rnn_hidden_new, next_actions

    @tf.function
    def soft_update(self, tau=0.005):
        for key in self.model_keys:
            for ep, tp in zip(self.actor_representation[key].variables,
                              self.target_actor_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_representation[key].variables,
                              self.target_critic_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.actor[key].variables, self.target_actor[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic[key].variables, self.target_critic[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)


class MADDPG_Policy(Independent_DDPG_Policy):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Optional[Dict[str, Module]],
                 critic_representation: Optional[Dict[str, Module]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
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

    @tf.function
    def Qpolicy(self, joint_observation: np.ndarray, joint_actions: np.ndarray,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (np.ndarray): The joint observations of the team.
            joint_actions (np.ndarray): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_eval: The evaluations of Q^policy.
        """
        rnn_hidden_new, q_eval = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs = {k: self.critic_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            rnn_hidden_new.update({k: (outputs[k]['rnn_hidden'], outputs[k]['rnn_cell']) for k in agent_list})
        else:
            outputs = {k: self.critic_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out = tf.repeat(tf.expand_dims(outputs[key]['state'], axis=1), self.n_agents, axis=1)
                    joint_rep_out = tf.reshape(joint_rep_out, [bs, seq_len, -1])
                else:
                    joint_rep_out = tf.repeat(tf.expand_dims(outputs[key]['state'], 1), self.n_agents, axis=1)
                    joint_rep_out = tf.reshape(joint_rep_out, [bs, -1])
                critic_in = tf.concat([joint_rep_out, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out = tf.reshape(outputs[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out = tf.reshape(outputs[key]['state'], [bs, -1])
                critic_in = joint_rep_out
            q_eval[key] = self.critic[key](critic_in)
        return rnn_hidden_new, q_eval

    @tf.function
    def Qtarget(self, joint_observation: np.ndarray, joint_actions: np.ndarray,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (np.ndarray): The joint observations of the team.
            joint_actions (np.ndarray): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_new, q_target = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs = {k: self.target_critic_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            rnn_hidden_new.update({k: (outputs[k]['rnn_hidden'], outputs[k]['rnn_cell']) for k in agent_list})
        else:
            outputs = {k: self.target_critic_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out = tf.repeat(tf.expand_dims(outputs[key]['state'], axis=1), self.n_agents, axis=1)
                    joint_rep_out = tf.reshape(joint_rep_out, [bs, seq_len, -1])
                else:
                    joint_rep_out = tf.repeat(tf.expand_dims(outputs[key]['state'], axis=1), self.n_agents, axis=1)
                    joint_rep_out = tf.reshape(joint_rep_out, [bs, -1])
                critic_in = tf.concat([joint_rep_out, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out = tf.reshape(outputs[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out = tf.reshape(outputs[key]['state'], [bs, -1])
                critic_in = joint_rep_out
            q_target[key] = self.target_critic[key](critic_in)
        return rnn_hidden_new, q_target


class MATD3_Policy(MADDPG_Policy, Module):
    def __init__(self,
                 action_space: Optional[Dict[str, Box]],
                 n_agents: int,
                 actor_representation: Optional[Dict[str, Module]],
                 critic_representation: Optional[Dict[str, Module]],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None,
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

        self.actor, self.target_actor, self.critic_A, self.critic_B = {}, {}, {}, {}
        self.target_critic_A, self.target_critic_B = {}, {}
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_A_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                       normalize, initialize, activation, activation_action)
            self.target_actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                              normalize, initialize, activation, activation_action)
            self.critic_A[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.critic_B[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic_A[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_critic_B[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            self.target_actor[key].set_weights(self.actor[key].get_weights())
            self.target_critic_A[key].set_weights(self.critic_A[key].get_weights())
            self.target_critic_B[key].set_weights(self.critic_B[key].get_weights())

    def critic_trainable_variables(self, key):
        return self.critic_A_representation[key].trainable_variables + self.critic_A[key].trainable_variables + \
               self.critic_B_representation[key].trainable_variables + self.critic_B[key].trainable_variables

    @tf.function
    def Qpolicy(self, joint_observation: np.ndarray, joint_actions: np.ndarray,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (np.ndarray): The joint observations of the team.
            joint_actions (np.ndarray): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            q_eval_A (Dict[Tensor]): The evaluations of Q^policy calculated by critic A.
            q_eval_B (Dict[Tensor]): The evaluations of Q^policy calculated by critic B.
            q_eval (Dict[Tensor]): The evaluations of Q^policy averaged by critic A and Critic B.
        """
        q_eval, q_eval_A, q_eval_B = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_A = {k: self.critic_A_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            outputs_B = {k: self.critic_B_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
        else:
            outputs_A = {k: self.critic_A_representation[k](critic_rep_in) for k in agent_list}
            outputs_B = {k: self.critic_B_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                joint_rep_out_A = tf.repeat(tf.expand_dims(outputs_A[key]['state'], axis=1), self.n_agents, axis=1)
                joint_rep_out_B = tf.repeat(tf.expand_dims(outputs_B[key]['state'], axis=1), self.n_agents, axis=1)
                if self.use_rnn:
                    joint_rep_out_A = tf.reshape(joint_rep_out_A, [bs, seq_len, -1])
                    joint_rep_out_B = tf.reshape(joint_rep_out_B, [bs, seq_len, -1])
                else:
                    joint_rep_out_A = tf.reshape(joint_rep_out_A, [bs, -1])
                    joint_rep_out_B = tf.reshape(joint_rep_out_B, [bs, -1])
                critic_in_A = tf.concat([joint_rep_out_A, agent_ids], axis=-1)
                critic_in_B = tf.concat([joint_rep_out_B, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_A = tf.reshape(outputs_A[key]['state'], [bs, seq_len, -1])
                    joint_rep_out_B = tf.reshape(outputs_B[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out_A = tf.reshape(outputs_A[key]['state'], [bs, -1])
                    joint_rep_out_B = tf.reshape(outputs_B[key]['state'], [bs, -1])
                critic_in_A = joint_rep_out_A
                critic_in_B = joint_rep_out_B
            q_eval_A[key] = self.critic_A[key](critic_in_A)
            q_eval_B[key] = self.critic_B[key](critic_in_B)
            q_eval[key] = (q_eval_A[key] + q_eval_B[key]) / 2.0

        return q_eval_A, q_eval_B, q_eval

    @tf.function
    def Qtarget(self, joint_observation: np.ndarray, joint_actions: np.ndarray,
                agent_ids: np.ndarray = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (np.ndarray): The joint observations of the team.
            joint_actions (np.ndarray): The joint actions of the team.
            agent_ids (Dict[np.ndarray]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[np.ndarray]]]): The hidden variables of the RNN.

        Returns:
            q_target (Dict[Tensor]): The evaluations of Q^target.
        """
        q_target = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = tf.concat([joint_observation, joint_actions], axis=-1)
        if self.use_rnn:
            outputs_A = {k: self.target_critic_A_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
            outputs_B = {k: self.target_critic_B_representation[k](critic_rep_in, *rnn_hidden[k]) for k in agent_list}
        else:
            outputs_A = {k: self.target_critic_A_representation[k](critic_rep_in) for k in agent_list}
            outputs_B = {k: self.target_critic_B_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                joint_rep_out_A = tf.repeat(tf.expand_dims(outputs_A[key]['state'], axis=1), self.n_agents, axis=1)
                joint_rep_out_B = tf.repeat(tf.expand_dims(outputs_B[key]['state'], axis=1), self.n_agents, axis=1)
                if self.use_rnn:
                    joint_rep_out_A = tf.reshape(joint_rep_out_A, [bs, seq_len, -1])
                    joint_rep_out_B = tf.reshape(joint_rep_out_B, [bs, seq_len, -1])
                else:
                    joint_rep_out_A = tf.reshape(joint_rep_out_A, [bs, -1])
                    joint_rep_out_B = tf.reshape(joint_rep_out_B, [bs, -1])
                critic_in_A = tf.concat([joint_rep_out_A, agent_ids], axis=-1)
                critic_in_B = tf.concat([joint_rep_out_B, agent_ids], axis=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_A = tf.reshape(outputs_A[key]['state'], [bs, seq_len, -1])
                    joint_rep_out_B = tf.reshape(outputs_B[key]['state'], [bs, seq_len, -1])
                else:
                    joint_rep_out_A = tf.reshape(outputs_A[key]['state'], [bs, -1])
                    joint_rep_out_B = tf.reshape(outputs_B[key]['state'], [bs, -1])
                critic_in_A = joint_rep_out_A
                critic_in_B = joint_rep_out_B
            q_target_A = self.target_critic_A[key](critic_in_A)
            q_target_B = self.target_critic_B[key](critic_in_B)
            q_target[key] = tf.math.minimum(q_target_A, q_target_B)

        return q_target

    @tf.function
    def soft_update(self, tau=0.005):
        for key in self.model_keys:
            for ep, tp in zip(self.actor_representation[key].variables,
                              self.target_actor_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_A_representation[key].variables,
                              self.target_critic_A_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_B_representation[key].variables,
                              self.target_critic_B_representation[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.actor[key].variables, self.target_actor[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_A[key].variables, self.target_critic_A[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
            for ep, tp in zip(self.critic_B[key].variables, self.target_critic_B[key].variables):
                tp.assign((1 - tau) * tp + tau * ep)
