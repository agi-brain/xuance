import numpy as np
from copy import deepcopy
from operator import itemgetter
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Union, Dict, List
from xuance.tensorflow.policies import CategoricalActorNet, ActorNet
from xuance.tensorflow.policies.core import CriticNet, BasicQhead
from xuance.tensorflow.utils import CategoricalDistribution
from xuance.tensorflow.representations import Basic_Identical
from xuance.tensorflow import tf, tk, Tensor, Module


class MAAC_Policy(Module):
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
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation_actor: dict,
                 representation_critic: dict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
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

    @tf.function
    def call(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
             avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
             rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of actor representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of actor representation.
            pi_dists (dict): The stochastic policy distributions.
        """
        rnn_hidden_new, pi_logits = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.actor_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_input = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_input = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_logits

    @tf.function
    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Get critic values via critic networks.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, values = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                critic_input = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                critic_input = outputs['state']

            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = tf.convert_to_tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAAC_Policy_Share(MAAC_Policy):
    """
    MAAC_Policy_Share: Multi-agent actor-critic Policy with categorical policies and shared representations.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The discrete action space.
        n_agents (int): The number of agents.
        representation (ModuleDict): A dict of representation modules.
        mixer (Module): The mixer module that mix together the individual values to the total value.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Module,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.is_continuous = False
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
        self.pi_dist = CategoricalDistribution(self.action_dim)

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        obs_shape = observation.shape
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            outputs_state = outputs['state']  # need to be improved
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            observation_reshape = tf.reshape(observation, [-1, obs_shape[-1]])
            outputs = self.representation(observation_reshape)
            outputs_state = tf.reshape(outputs['state'], obs_shape[:-1] + self.representation_info_shape['state'])
            rnn_hidden = None
        actor_critic_input = tf.concat([outputs_state, agent_ids], axis=-1)
        act_logits = self.actor(actor_critic_input)
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            avail_actions = tf.convert_to_tensor(kwargs['avail_actions'])
            act_logits[avail_actions == 0] = -1e10
            self.pi_dist.set_param(logits=act_logits)
        else:
            self.pi_dist.set_param(logits=act_logits)

        values_independent = self.critic(actor_critic_input)
        if self.use_rnn:
            pass  # to do
        else:
            values_tot = values_independent if self.mixer is None else self.value_tot(values_independent,
                                                                                      global_state=kwargs['state'])
            values_tot = tf.repeat(tf.expand_dims(values_tot, 1), repeats=self.n_agents, axis=1)

        return rnn_hidden, self.pi_dist, values_tot

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = tf.convert_to_tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)

    def trainable_param(self):
        params = self.actor.trainable_variables + self.critic.trainable_variables
        if self.mixer is not None:
            params += self.mixer.trainable_variables
        if self.identical_rep:
            return params
        else:
            return params + self.representation.trainable_variables


class COMA_Policy(Module):
    """
    COMA_Policy: Counterfactual Multi-Agent Actor-Critic Policy with categorical distributions.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The discrete action space.
        n_agents (int): The number of agents.
        representation_actor (ModuleDict): A dict of representation modules for each agent's actor.
        representation_critic (ModuleDict): A dict of representation modules for each agent's critic.
        mixer (Module): The mixer module that mix together the individual values to the total value.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: Union[Module, dict],
                 representation_critic: Union[Module, dict],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(COMA_Policy, self).__init__()
        self.is_continuous = False
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic
        self.target_critic_representation = deepcopy(self.critic_representation)

        # create actor
        self.n_actions = {k: space.n for k, space in self.action_space.items()}
        self.actor = {}
        for key in self.model_keys:
            dim_actor_input = self.actor_representation[key].output_shapes['state'][0]
            if self.use_parameter_sharing:
                dim_actor_input += self.n_agents
            self.actor[key] = CategoricalActorNet(dim_actor_input, self.n_actions[key], actor_hidden_size,
                                                  normalize, initializer, activation)

        dim_input_critic = kwargs['dim_global_state']
        dim_input_critic += self.critic_representation[self.model_keys[0]].output_shapes['state'][0]
        dim_input_critic += sum(self.n_actions.values())
        dim_input_critic += self.n_agents
        self.n_actions_max = max(self.n_actions.values())
        self.critic = BasicQhead(dim_input_critic, self.n_actions_max,
                                 critic_hidden_size, normalize, initializer, activation)
        self.target_critic = BasicQhead(dim_input_critic, self.n_actions_max,
                                        critic_hidden_size, normalize, initializer, activation)
        self.target_critic.set_weights(self.critic.get_weights())

    @property
    def parameters_actor(self):
        params = []
        if isinstance(self.actor_representation[self.model_keys[0]], Basic_Identical):
            for k in self.model_keys:
                params.extend(self.actor_representation[k].trainable_variables)
        else:
            for k in self.model_keys:
                params.extend(self.actor[k].trainable_variables + self.actor[k].trainable_variables)
        return params

    @property
    def parameters_critic(self):
        params = self.critic.trainable_variables
        if not isinstance(self.critic_representation[self.model_keys[0]], Basic_Identical):
            for k in self.model_keys:
                params += self.critic_representation[k].trainable_variables
        return params

    @tf.function
    def call(self, observation: Union[np.ndarray, dict], agent_ids: Union[np.ndarray, dict] = None,
             avail_actions: Union[np.ndarray, dict] = None, agent_key: str = None,
             rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, epsilon=0.0, test_mode=False, **kwargs):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of actor representation.
            epsilon: The epsilon.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of actor representation.
            act_probs (dict): The probabilities of the actions.
        """
        rnn_hidden_new, pi_logits, act_probs = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.actor_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_input = tf.concat([outputs['state'], agent_ids], axis=-1)
            else:
                actor_input = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)
            act_probs[key] = tf.nn.softmax(pi_logits[key], axis=-1)

            if not test_mode:
                act_probs[key] = (1 - epsilon) * act_probs[key] + epsilon * 1 / self.n_actions[key]

        return rnn_hidden_new, act_probs

    def get_values(self, state: Tensor, observation: Dict[str, Tensor], actions: Dict[str, Tensor],
                   agent_ids: Tensor = None, rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, target=False):
        """
        Get evaluated critic values.

        Parameters:
            state: Tensor: The global state.
            observation (Dict[str, Tensor]): The input observations for the policies.
            actions (Dict[str, Tensor]): The input actions.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The RNN hidden states of critic representation.
            target: If to use target critic network to calculate the critic values.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, critic_input = {}, {}
        batch_size = state.shape[0]
        seq_len = state.shape[1] if self.use_rnn else 1
        critic_inputs = []

        if self.use_rnn:
            critic_inputs.append(tf.tile(tf.expand_dims(state, -2), [1, 1, self.n_agents, 1]))  # batch * T * N * dim_S)
        else:
            critic_inputs.append(tf.tile(tf.expand_dims(state, -2), [1, self.n_agents, 1]))  # batch * N * dim_S

        obs_rep = {}
        for key in self.model_keys:
            if self.use_rnn:
                if target:
                    outputs = self.target_critic_representation[key](observation[key], *rnn_hidden[key])
                else:
                    outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                if target:
                    outputs = self.target_critic_representation[key](observation[key])
                else:
                    outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]
            obs_rep[key] = outputs['state']

        agent_mask = tf.expand_dims((1 - tf.eye(self.n_agents, dtype=tf.float32)), axis=-1)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            agent_mask = tf.reshape(tf.tile(agent_mask, [1, 1, self.n_actions[key]]), [self.n_agents, -1])
            agent_mask = tf.expand_dims(agent_mask, 0)
            if self.use_rnn:
                actions_input = actions[key].reshape(batch_size, seq_len, 1, -1).repeat(1, 1, self.n_agents, 1)
                critic_inputs.append(obs_rep[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
                critic_inputs.append(actions_input * agent_mask.unsqueeze(0))
                critic_inputs.append(agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
            else:
                actions_input = tf.tile(tf.reshape(actions[key], [batch_size, 1, -1]), [1, self.n_agents, 1])
                critic_inputs.append(tf.reshape(obs_rep[key], [batch_size, self.n_agents, -1]))
                critic_inputs.append(actions_input * agent_mask)
                critic_inputs.append(tf.reshape(agent_ids, [batch_size, self.n_agents, -1]))
            critic_inputs = tf.concat(critic_inputs, axis=-1)
        else:
            agent_mask = tf.concat([agent_mask[i].repeat(1, self.n_actions[k])
                                    for i, k in enumerate(self.model_keys)], axis=-1).unsqueeze(0)
            if self.use_rnn:
                agent_mask = agent_mask.unsqueeze(1)
                actions_input = tf.concat(itemgetter(*self.model_keys)(actions),
                                          axis=-1).unsqueeze(-2).repeat(1, 1, self.n_agents, 1)  # batch * T * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2)
            else:
                actions_input = tf.concat(itemgetter(*self.model_keys)(actions),
                                          axis=-1).unsqueeze(1).repeat(1, self.n_agents, 1)  # batch_size * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, -1)  # batch_size * N * N
            critic_inputs.append(tf.stack(itemgetter(*self.model_keys)(obs_rep), axis=-2))
            critic_inputs.append(actions_input * agent_mask)
            critic_inputs.append(agent_ids)
            critic_inputs = tf.concat(critic_inputs, axis=-1)

        values = self.target_critic(critic_inputs) if target else self.critic(critic_inputs)
        return rnn_hidden_new, values

    def copy_target(self):
        for key in self.model_keys:
            self.target_critic_representation[key].set_weights(self.critic_representation[key].get_weights())
        self.target_critic.set_weights(self.critic.get_weights())


class MeanFieldActorCriticPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[tk.layers.Layer] = None,
                 initializer: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.is_continuous = False
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim,
                                  actor_hidden_size, normalize, initializer, activation)
        self.critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim,
                                    critic_hidden_size, normalize, initializer, activation)
        self.trainable_param = self.actor_net.trainable_variables + self.critic_net.trainable_variables
        self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False
        self.pi_dist = CategoricalDistribution(self.action_dim)

    @tf.function
    def call(self, inputs: Union[np.ndarray, dict], **kwargs):
        observations = inputs['obs']
        IDs = inputs['ids']
        outputs = self.representation(observations)
        input_actor = tf.concat([outputs['state'], IDs], axis=-1)
        act_logits = self.actor_net(input_actor)
        self.pi_dist.set_param(logits=act_logits)
        return outputs, self.pi_dist

    def trainable_param(self):
        params = self.actor_net.trainable_variables + self.critic_net.trainable_variables
        if self.identical_rep:
            return params
        else:
            return params + self.representation.trainable_variables

    def critic(self, observation: Tensor, actions_mean: Tensor, agent_ids: Tensor):
        outputs = self.representation(observation)
        critic_in = tf.concat([outputs['state'], actions_mean, agent_ids], axis=-1)
        critic_out = tf.expand_dims(self.critic_net(critic_in), -1)
        return critic_out
