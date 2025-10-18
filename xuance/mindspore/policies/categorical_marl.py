import mindspore as ms
from copy import deepcopy
from operator import itemgetter
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.mindspore.policies import CategoricalActorNet, ActorNet
from xuance.mindspore.policies.core import CriticNet, BasicQhead
from xuance.mindspore.policies import VDN_mixer
from xuance.mindspore.utils import ModuleType
from xuance.mindspore.representations import Basic_Identical, Basic_MLP
from xuance.mindspore import Tensor, Module, ModuleDict, ops


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
        initializer (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
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
        self.actor, self.critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initializer, activation)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initializer, activation)
            # update parameters name
            self.actor_representation[key].update_parameters_name(key + '_rep_actor_')
            self.critic_representation[key].update_parameters_name(key + '_rep_critic_')
            self.actor[key].update_parameters_name(key + '_actor_')
            self.critic[key].update_parameters_name(key + '_critic_')

        self.mixer = mixer

    @property
    def parameters_model(self):
        parameters = self.actor_representation.trainable_params() + self.actor.trainable_params() + \
                     self.critic_representation.trainable_params() + self.critic.trainable_params()
        if self.mixer is not None:
            parameters = parameters + self.mixer.trainable_params()
        return parameters

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

    def construct(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
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
                actor_input = ops.cat([outputs, agent_ids], axis=-1)
            else:
                actor_input = outputs

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden, pi_logits

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
                critic_input = ops.cat([outputs, agent_ids], axis=-1)
            else:
                critic_input = outputs

            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = global_state
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
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
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
                              actor_hidden_size, normalize, initialize, kwargs['gain'], activation)
        self.critic = CriticNet(self.representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                normalize, initialize, activation)
        self.mixer = mixer
        self.identical_rep = True if isinstance(self.representation, Basic_Identical) else False

    def construct(self, inputs: Union[Tensor, dict], *rnn_hidden, **kwargs):
        observation = inputs['obs']
        agent_ids = inputs['ids']
        obs_shape = observation.shape
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            outputs_state = outputs['state']  # need to be improved
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            observation_reshape = observation.reshape([-1, obs_shape[-1]])
            outputs = self.representation(observation_reshape)
            outputs_state = outputs['state'].reshape(obs_shape[:-1] + self.representation_info_shape['state'])
            rnn_hidden = None
        actor_critic_input = ops.cat([outputs_state, agent_ids], axis=-1)
        act_logits = self.actor(actor_critic_input)
        if ('avail_actions' in kwargs.keys()) and (kwargs['avail_actions'] is not None):
            avail_actions = Tensor(kwargs['avail_actions'])
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
            values_tot = values_tot.unsqueeze(1).repeat(1, self.n_agents, 1)

        return rnn_hidden, self.pi_dist, values_tot

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = Tensor(global_state)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)

    def trainable_param(self):
        params = self.actor.trainable_variables + self.critic.trainable_variables
        if self.mixer is not None:
            params += self.mixer.trainable_variables
        if self.identical_rep:
            return params
        else:
            return params + self.representation.trainable_variables


class CommNet_Policy(MAAC_Policy):

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(CommNet_Policy, self).__init__(action_space=action_space, n_agents=n_agents, representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer, actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize, initialize=initialize, activation=activation,
                                             use_distributed_training=use_distributed_training, **kwargs)
        self.communicator = kwargs['communicator']
        self.agent_keys = kwargs['agent_keys']
        self.comm_passes = kwargs['comm_passes']

    @property
    def parameters_model(self):
        parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters()) + list(
            self.critic_representation.parameters()) + list(self.critic.parameters()) + list(self.communicator.parameters())
        if self.mixer is not None:
            parameters += list(self.mixer.parameters())
        return parameters

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        seq_length = observation['agent_0'].shape[1]
        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}
        observation = {k: self.communicator[k].obs_encode(observation[k]) for k in agent_list}
        actor_inputs = {k: [] for k in agent_list}
        for i in range(seq_length):
            alive_ally_i = {k: alive_ally[k][:, i:i + 1, :] for k in self.agent_keys}
            observation_i = {k: observation[k][:, i:i + 1, :] for k in agent_list}
            msg_send = {k: rnn_hidden[k][0].transpose(0, 1) for k in self.model_keys}
            for _ in range(self.comm_passes):
                msg_receive = {k: self.communicator[k](observation_i[k], msg_send, alive_ally_i) for k in self.model_keys}
                msg_send = {k: observation_i[k] + msg_receive[k] for k in self.model_keys}
            observation_i = msg_send
            for key in agent_list:
                if self.use_rnn:
                    outputs = self.actor_representation[key](observation_i[key], *rnn_hidden[key])
                    rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.actor_representation[key](observation[key])
                    rnn_hidden_new[key] = [None, None]

                if self.use_parameter_sharing:
                    agent_ids_i = agent_ids[:, i:i + 1, :]
                    actor_input = torch.concat([outputs['state'], agent_ids_i], dim=-1)
                else:
                    actor_input = outputs['state']
                actor_inputs[key].append(actor_input)
            rnn_hidden = deepcopy(rnn_hidden_new)
        for key in agent_list:
            actor_input = torch.cat(actor_inputs[key], dim=1)
            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_dists

    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, values = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size, seq_length = observation['agent_0'].shape[0], observation['agent_0'].shape[1]
        observation = {k: self.communicator[k].obs_encode(observation[k]) for k in agent_list}
        critic_inputs = {k: [] for k in agent_list}
        for i in range(seq_length):
            alive_ally_i = {k: alive_ally[k][:, i:i + 1, :] for k in self.agent_keys}
            observation_i = {k: observation[k][:, i:i + 1, :] for k in agent_list}
            msg_send = {k: rnn_hidden[k][0].transpose(0, 1) for k in self.model_keys}
            for _ in range(self.comm_passes):
                msg_receive = {k: self.communicator[k](observation_i[k], msg_send, alive_ally_i) for k in self.model_keys}
                msg_send = {k: observation_i[k] + msg_receive[k][0] for k in self.model_keys}
            observation_i = msg_send
            for key in agent_list:
                if self.use_rnn:
                    outputs = self.critic_representation[key](observation_i[key], *rnn_hidden[key])
                    rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.critic_representation[key](observation[key])
                    rnn_hidden_new[key] = [None, None]

                if self.use_parameter_sharing:
                    agent_ids_i = agent_ids[:, i:i + 1, :]
                    critic_input = torch.concat([outputs['state'], agent_ids_i], dim=-1)
                else:
                    critic_input = outputs['state']
                critic_inputs[key].append(critic_input)
            rnn_hidden = deepcopy(rnn_hidden_new)

        for key in agent_list:
            critic_input = torch.cat(critic_inputs[key], dim=1)
            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values


class IC3Net_Policy(CommNet_Policy):

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(IC3Net_Policy, self).__init__(action_space=action_space, n_agents=n_agents,
                                             representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer,
                                             actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize,
                                             initialize=initialize, activation=activation,
                                             use_distributed_training=use_distributed_training, **kwargs)

        self.config = kwargs['config']
        self.gate = {k: self.communicator[k].create_mlp(self.config.recurrent_hidden_size, self.config.gate_hidden_size, 2, nn.LeakyReLU())
                     for k in self.model_keys}

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        seq_length = observation['agent_0'].shape[1]
        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}
        observation = {k: self.communicator[k].obs_encode(observation[k]) for k in agent_list}
        actor_inputs = {k: [] for k in agent_list}
        gate_log_probs = {k: [] for k in agent_list}
        for i in range(seq_length):
            alive_ally_i = {k: alive_ally[k][:, i:i + 1, :] for k in self.agent_keys}
            observation_i = {k: observation[k][:, i:i + 1, :] for k in agent_list}
            msg_send = {k: rnn_hidden[k][0].transpose(0, 1) for k in self.model_keys}
            for comm_time in range(self.comm_passes):
                # calculate gate_control
                gate_prob = {k: self.gate[k](msg_send[k]) for k in agent_list}
                gate_dist = {k: Categorical(logits=gate_prob[k]) for k in agent_list}
                gate_control = {k: gate_dist[k].sample() for k in agent_list}
                gate_log_prob = {k: gate_dist[k].log_prob(gate_control[k]) for k in agent_list}
                comm_out = {k: self.communicator[k](observation_i[k], msg_send, alive_ally_i, gate_control) for k in self.model_keys}
                msg_send = {k: observation_i[k] + comm_out[k] for k in self.model_keys}
                if comm_time == self.comm_passes - 1:
                    for k in agent_list:
                        gate_log_probs[k].append(gate_log_prob[k])
            observation_i = msg_send
            for key in agent_list:
                if self.use_rnn:
                    outputs = self.actor_representation[key](observation_i[key], *rnn_hidden[key])
                    rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.actor_representation[key](observation[key])
                    rnn_hidden_new[key] = [None, None]

                if self.use_parameter_sharing:
                    agent_ids_i = agent_ids[:, i:i + 1, :]
                    actor_input = torch.concat([outputs['state'], agent_ids_i], dim=-1)
                else:
                    actor_input = outputs['state']
                actor_inputs[key].append(actor_input)
            rnn_hidden = deepcopy(rnn_hidden_new)
        for key in agent_list:
            actor_input = torch.cat(actor_inputs[key], dim=1)
            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        gate_log_probs = {k: torch.cat(gate_log_probs[k], dim=1) for k in self.model_keys}
        return rnn_hidden_new, pi_dists, gate_log_probs

    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, values = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size, seq_length = observation['agent_0'].shape[0], observation['agent_0'].shape[1]
        observation = {k: self.communicator[k].obs_encode(observation[k]) for k in agent_list}
        critic_inputs = {k: [] for k in agent_list}
        for i in range(seq_length):
            alive_ally_i = {k: alive_ally[k][:, i:i + 1, :] for k in self.agent_keys}
            observation_i = {k: observation[k][:, i:i + 1, :] for k in agent_list}
            msg_send = {k: rnn_hidden[k][0].transpose(0, 1) for k in self.model_keys}
            for comm_time in range(self.comm_passes):
                # calculate gate_control
                gate_prob = {k: self.gate[k](msg_send[k]) for k in agent_list}
                gate_dist = {k: Categorical(logits=gate_prob[k]) for k in agent_list}
                gate_control = {k: gate_dist[k].sample() for k in agent_list}
                comm_out = {k: self.communicator[k](observation_i[k], msg_send, alive_ally_i, gate_control) for k in
                            self.model_keys}
                msg_send = {k: observation_i[k] + comm_out[k] for k in self.model_keys}
            observation_i = msg_send
            for key in agent_list:
                if self.use_rnn:
                    outputs = self.critic_representation[key](observation_i[key], *rnn_hidden[key])
                    rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.critic_representation[key](observation[key])
                    rnn_hidden_new[key] = [None, None]

                if self.use_parameter_sharing:
                    agent_ids_i = agent_ids[:, i:i + 1, :]
                    critic_input = torch.concat([outputs['state'], agent_ids_i], dim=-1)
                else:
                    critic_input = outputs['state']
                critic_inputs[key].append(critic_input)
            rnn_hidden = deepcopy(rnn_hidden_new)

        for key in agent_list:
            critic_input = torch.cat(critic_inputs[key], dim=1)
            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values


class TarMAC_Policy(IC3Net_Policy):

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(TarMAC_Policy, self).__init__(action_space=action_space, n_agents=n_agents,
                                             representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer,
                                             actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize,
                                             initialize=initialize, activation=activation,
                                             use_distributed_training=use_distributed_training, **kwargs)

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        seq_length = observation['agent_0'].shape[1]
        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}
        observation = {k: self.communicator[k].obs_encode(observation[k]) for k in agent_list}
        actor_inputs = {k: [] for k in agent_list}
        gate_log_probs = {k: [] for k in agent_list}
        for i in range(seq_length):
            alive_ally_i = {k: alive_ally[k][:, i:i + 1, :] for k in self.agent_keys}
            observation_i = {k: observation[k][:, i:i + 1, :] for k in agent_list}
            msg_send = {k: rnn_hidden[k][0].transpose(0, 1) for k in self.model_keys}
            for comm_time in range(self.comm_passes):
                # calculate gate_control
                gate_prob = {k: self.gate[k](msg_send[k]) for k in agent_list}
                gate_dist = {k: Categorical(logits=gate_prob[k]) for k in agent_list}
                gate_control = {k: gate_dist[k].sample() for k in agent_list}
                gate_log_prob = {k: gate_dist[k].log_prob(gate_control[k]) for k in agent_list}
                comm_out = {k: self.communicator[k](observation_i[k], msg_send, alive_ally_i, gate_control) for k in self.model_keys}
                msg_send = {k: observation_i[k] + comm_out[k] for k in self.model_keys}
                if comm_time == self.comm_passes - 1:
                    for k in agent_list:
                        gate_log_probs[k].append(gate_log_prob[k])
            observation_i = msg_send
            for key in agent_list:
                if self.use_rnn:
                    outputs = self.actor_representation[key](observation_i[key], *rnn_hidden[key])
                    rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                else:
                    outputs = self.actor_representation[key](observation[key])
                    rnn_hidden_new[key] = [None, None]

                if self.use_parameter_sharing:
                    agent_ids_i = agent_ids[:, i:i + 1, :]
                    actor_input = torch.concat([outputs['state'], agent_ids_i], dim=-1)
                else:
                    actor_input = outputs['state']
                actor_inputs[key].append(actor_input)
            rnn_hidden = deepcopy(rnn_hidden_new)
        for key in agent_list:
            actor_input = torch.cat(actor_inputs[key], dim=1)
            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        gate_log_probs = {k: torch.cat(gate_log_probs[k], dim=1) for k in self.model_keys}
        return rnn_hidden_new, pi_dists, gate_log_probs


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
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
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
        self.actor = ModuleDict()
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
        self.target_critic = deepcopy(self.critic)
        # update parameters name
        for key in self.model_keys:
            self.critic_representation[key].update_parameters_name(key + '_critic_rep_')
        self.critic.update_parameters_name('_critic_')

    @property
    def parameters_actor(self):
        if isinstance(self.actor_representation[self.model_keys[0]], Basic_Identical):
            params = self.actor.trainable_params()
        else:
            params = self.actor_representation.trainable_params() + self.actor.trainable_params()
        return params

    @property
    def parameters_critic(self):
        if isinstance(self.critic_representation[self.model_keys[0]], Basic_Identical):
            params = self.critic.trainable_params()
        else:
            params = self.critic_representation.trainable_params() + self.critic.trainable_params()
        return params

    def construct(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                  avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                  rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, **kwargs):
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
            pi_logits (dict): The output of the actors.
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
                actor_input = ops.cat([outputs, agent_ids], axis=-1)
            else:
                actor_input = outputs

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)

        return rnn_hidden_new, pi_logits

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
            critic_inputs.append(ops.repeat_elements(state.unsqueeze(-2),
                                                     rep=self.n_agents, axis=2))  # batch * T * N * dim_S
        else:
            critic_inputs.append(ops.repeat_elements(state.unsqueeze(-2),
                                                     rep=self.n_agents, axis=1))  # batch * N * dim_S

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
            obs_rep[key] = outputs

        agent_mask = (1 - ops.eye(self.n_agents, dtype=ms.float32)).unsqueeze(-1)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            agent_mask = ops.repeat_elements(agent_mask, rep=int(self.n_actions[key]),
                                             axis=2).reshape(self.n_agents, -1).unsqueeze(0)
            if self.use_rnn:
                actions_input = ops.repeat_elements(actions[key].reshape(batch_size, seq_len, 1, -1),
                                                    rep=self.n_agents, axis=2)
                critic_inputs.append(obs_rep[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
                critic_inputs.append(actions_input * agent_mask.unsqueeze(0))
                critic_inputs.append(agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
            else:
                actions_input = ops.repeat_elements(actions[key].reshape(batch_size, 1, -1),
                                                    rep=self.n_agents, axis=1)
                critic_inputs.append(obs_rep[key].reshape(batch_size, self.n_agents, -1))
                critic_inputs.append(actions_input * agent_mask)
                critic_inputs.append(agent_ids.reshape(batch_size, self.n_agents, -1))
            critic_inputs = ops.cat(critic_inputs, axis=-1)
        else:
            agent_mask = ops.cat([ops.repeat_elements(agent_mask[i], rep=self.n_actions[k], axis=1)
                                  for i, k in enumerate(self.model_keys)], axis=-1).unsqueeze(0)
            if self.use_rnn:
                agent_mask = agent_mask.unsqueeze(1)
                actions_input = ops.repeat_elements(ops.cat(itemgetter(*self.model_keys)(actions),
                                                            axis=-1).unsqueeze(-2),
                                                    rep=self.n_agents, axis=2)  # batch * T * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2)
            else:
                actions_input = ops.repeat_elements(ops.cat(itemgetter(*self.model_keys)(actions),
                                                            axis=-1).unsqueeze(1),
                                                    rep=self.n_agents, axis=1)  # batch_size * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, -1)  # batch_size * N * N
            critic_inputs.append(ops.stack(itemgetter(*self.model_keys)(obs_rep), axis=-2))
            critic_inputs.append(actions_input * agent_mask)
            critic_inputs.append(agent_ids)
            critic_inputs = ops.cat(critic_inputs, axis=-1)

        values = self.target_critic(critic_inputs) if target else self.critic(critic_inputs)
        return rnn_hidden_new, values

    def copy_target(self):
        for ep, tp in zip(self.critic_representation.trainable_params(),
                          self.target_critic_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            tp.assign_value(ep)


class MeanFieldActorCriticPolicy(Module):
    """Mean-field actor-critic policy.

    This policy maintains separate actor and critic networks for each agent type (model key),
    embeds the mean action of neighboring agents, and produces Boltzmann policies.

    Args:
        action_space (Discrete): A mapping from model keys to discrete action spaces.
        n_agents (int): Total number of agents in the environment.
        representation_actor (Optional[Dict[str, Module]]): Actor state encoder modules for each model key.
        representation_critic (Optional[Dict[str, Module]]): Critic state encoder modules for each model key.
        actor_hidden_size (Sequence[int], optional): Hidden layer sizes for actor networks.
        critic_hidden_size (Sequence[int], optional): Hidden layer sizes for critic networks.
        normalize (Optional[tk.layers.Layer]): Normalization layer to apply after each hidden layer.
        initialize (Optional[tk.initializers.Initializer]): Weight initialization function.
        activation (Optional[tk.layers.Layer]): Activation function class for hidden layers.
        use_distributed_training (bool): If True, wrap components in DistributedDataParallel.
        **kwargs: Additional keyword arguments:
            use_parameter_sharing (bool): Whether to share parameters across agent types.
            model_keys (List[str]): Keys identifying different agent types.
            rnn (str): RNN type, e.g., "LSTM" or "GRU".
            use_rnn (bool): Flag indicating whether to include RNN layers.
            action_embedding_hidden_size (Sequence[int]): Hidden sizes for action mean embedding.
            temperature (float): Temperature parameter for Boltzmann policy.
    """

    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.is_continuous = False
        self.action_space = action_space
        self.n_agents = n_agents
        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = int(max(self.n_actions_list))
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic

        self.dim_input_critic, self.n_actions = {}, {}
        self.action_mean_embedding = ModuleDict()
        self.actor, self.critic = ModuleDict(), ModuleDict()
        dim_action_embedding = self.n_actions_max + self.n_agents if self.use_parameter_sharing else self.n_actions_max
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0],
                n_agents, )
            dim_critic_in += kwargs['action_embedding_hidden_size'][-1]

            self.action_mean_embedding[key] = Basic_MLP((dim_action_embedding,),
                                                        kwargs['action_embedding_hidden_size'],
                                                        normalize, initialize, activation)
            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initialize, activation)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation)
            # update parameters name
            self.actor_representation[key].update_parameters_name(key + '_actor_rep_')
            self.critic_representation[key].update_parameters_name(key + '_critic_rep_')
            self.action_mean_embedding[key].update_parameters_name(key + '_act_embedding_')
            self.actor[key].update_parameters_name(key + '_actor_')
            self.critic[key].update_parameters_name(key + '_critic_')
        self.temperature = kwargs['temperature']

    @property
    def parameters_model(self):
        params = self.actor.trainable_params() + self.critic.trainable_params()
        if not isinstance(self.actor_representation[self.model_keys[0]], Basic_Identical):
            params += self.actor_representation.trainable_params()
        if not isinstance(self.critic_representation[self.model_keys[0]], Basic_Identical):
            params += self.critic_representation.trainable_params()
        return params

    def construct(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                  avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                  rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, **kwargs):
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
        rnn_hidden_new, pi_logits, pi_dists = {}, {}, {}
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
                actor_input = ops.cat([outputs, agent_ids], axis=-1)
            else:
                actor_input = outputs

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_logits

    def get_mean_actions(self, actions: Dict[str, Tensor],
                         agent_mask_tensor: Tensor, batch_size: int):
        """Compute mean one-hot action vectors of each agent's neighbors.

        For each batch and agent, exclude the agent's own action and average the one-hot
        action encodings of its alive neighbors.

        Args:
            actions (Dict[str, Tensor]): Mapping from model keys to chosen action indices of shape [batch_size * n_agents].
            agent_mask_tensor (Tensor): Binary mask of shape [batch_size, n_agents] indicating alive (1) or dead (0) agents.
            batch_size (int): Number of samples in the batch.

        Returns:
            Tensor: Mean one-hot action tensor of shape [batch_size, n_agents, n_actions_max].
        """
        if self.use_parameter_sharing:
            actions_tensor = actions[self.model_keys[0]].reshape([-1, self.n_agents])
        else:
            actions_tensor = ops.stack(itemgetter(*self.model_keys)(actions), axis=-1).reshape([-1, self.n_agents])
        actions_onehot = ops.one_hot(actions_tensor, depth=self.n_actions_max)

        # count alive neighbors
        _eyes = ops.repeat_elements(ops.eye(self.n_agents).unsqueeze(0), rep=batch_size, axis=0)
        agent_mask_diagonal = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1), rep=self.n_agents, axis=-1) * _eyes
        agent_mask_neighbors = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1),
                                                   rep=self.n_agents, axis=-1) - agent_mask_diagonal
        agent_alive_neighbors = agent_mask_neighbors.sum(axis=-1, keepdims=True)

        # calculate mean actions of each agent's neighbors
        agent_mask_repeat = ops.repeat_elements(agent_mask_tensor.unsqueeze(-1), rep=self.n_actions_max, axis=-1)
        actions_onehot = actions_onehot * agent_mask_repeat
        actions_sum = ops.repeat_elements(actions_onehot.sum(axis=-2, keepdims=True),
                                          rep=self.n_agents, axis=1)
        actions_neighbors_sum = actions_sum - actions_onehot  # Sum of other agents' actions.
        actions_mean_masked = actions_neighbors_sum * agent_mask_repeat / agent_alive_neighbors
        return actions_mean_masked

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor network and critic networks.

        Parameters:
            dim_action: The dimension of actions.
            dim_actor_rep: The dimension of the output of actor representation.
            dim_action_max: The maximum dimension of the output of actor
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

    def get_values(self, observation: Dict[str, Tensor],
                   actions_mean: Dict[str, Tensor] = None,
                   agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Get critic values via critic networks.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            actions_mean (Dict[str, Tensor]): The mean actions of each agent's neighbors.
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
                action_embedding_input = ops.cat([actions_mean[key], agent_ids], axis=-1)
                act_embedding = self.action_mean_embedding[key](action_embedding_input)
                critic_input = ops.cat([outputs, act_embedding, agent_ids], axis=-1)
            else:
                act_embedding = self.action_mean_embedding[key](actions_mean[key])
                critic_input = ops.cat([outputs, act_embedding], axis=-1)

            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values


class Basic_ISAC_Policy(Module):
    """
    Basic_ISAC_Policy: The basic policy for independent soft actor-critic.

    Args:
        action_space (Box): The continuous action space.
        n_agents (int): The number of agents.
        actor_representation (ModuleDict): A dict of representation modules for each agent's actor.
        critic_representation (ModuleDict): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 actor_representation: ModuleDict,
                 critic_representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()


class MASAC_Policy(Basic_ISAC_Policy):
    """
    Basic_ISAC_Policy: The basic policy for independent soft actor-critic.

    Args:
        action_space (Box): The continuous action space.
        n_agents (int): The number of agents.
        actor_representation (ModuleDict): A dict of representation modules for each agent's actor.
        critic_representation (ModuleDict): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 actor_representation: ModuleDict,
                 critic_representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation,
                                           use_distributed_training, **kwargs)

