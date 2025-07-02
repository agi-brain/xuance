import os
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from operator import itemgetter
from gymnasium.spaces import Discrete
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.torch.policies import CategoricalActorNet, ActorNet
from xuance.torch.policies.core import CriticNet, BasicQhead
from xuance.torch.utils import ModuleType
from xuance.torch.representations import Basic_MLP
from xuance.torch import Tensor, Module, ModuleDict, DistributedDataParallel
from .core import CategoricalActorNet_SAC as Actor_SAC


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with categorical policies.

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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
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
                                                  normalize, initialize, activation, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)

        self.mixer = mixer

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_representation[key]._get_name() != "Basic_Identical":
                    self.critic_representation[key] = DistributedDataParallel(self.critic_representation[key],
                                                                              device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic[key] = DistributedDataParallel(module=self.critic[key], device_ids=[self.rank])
            if self.mixer is not None:
                self.mixer = DistributedDataParallel(module=self.mixer, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters()) + list(
            self.critic_representation.parameters()) + list(self.critic.parameters())
        if self.mixer is not None:
            parameters += list(self.mixer.parameters())
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

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
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
        rnn_hidden_new, pi_dists = {}, {}
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
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_dists

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
                critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                critic_input = outputs['state']

            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
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
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.representation = representation

        self.dim_input_critic, self.n_actions = {}, {}
        self.actor, self.critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_rep = self.representation[key].output_shapes['state'][0]
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key], dim_rep, dim_rep, n_agents)

            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initialize, activation, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)

        self.mixer = mixer

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.representation[key]._get_name() != "Basic_Identical":
                    self.representation[key] = DistributedDataParallel(self.representation[key], device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic[key] = DistributedDataParallel(module=self.critic[key], device_ids=[self.rank])
            if self.mixer is not None:
                self.mixer = DistributedDataParallel(module=self.mixer, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters = list(self.representation.parameters()) + list(self.actor.parameters()) + \
                     list(self.critic.parameters())
        if self.mixer is not None:
            parameters += list(self.mixer.parameters())
        return parameters

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
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
        rnn_hidden_new, pi_dists, values = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_dists

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
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                critic_input = outputs['state']

            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values

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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(CommNet_Policy, self).__init__(action_space=action_space, n_agents=n_agents, representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer, actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize, initialize=initialize, activation=activation,
                                             device=device, use_distributed_training=use_distributed_training, **kwargs)
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(IC3Net_Policy, self).__init__(action_space=action_space, n_agents=n_agents,
                                             representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer,
                                             actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize,
                                             initialize=initialize, activation=activation,
                                             device=device, use_distributed_training=use_distributed_training, **kwargs)

        self.config = kwargs['config']
        self.gate = {k: self.communicator[k].create_mlp(self.config.recurrent_hidden_size, self.config.gate_hidden_size, 2, nn.LeakyReLU(), self.device)
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(TarMAC_Policy, self).__init__(action_space=action_space, n_agents=n_agents,
                                             representation_actor=representation_actor,
                                             representation_critic=representation_critic, mixer=mixer,
                                             actor_hidden_size=actor_hidden_size,
                                             critic_hidden_size=critic_hidden_size, normalize=normalize,
                                             initialize=initialize, activation=activation,
                                             device=device, use_distributed_training=use_distributed_training, **kwargs)

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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: Module,
                 representation_critic: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(COMA_Policy, self).__init__()
        self.device = device
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
            self.actor[key] = ActorNet(dim_actor_input, self.n_actions[key], actor_hidden_size,
                                       normalize, initialize, activation, None, device)

        dim_input_critic = kwargs['dim_global_state']
        dim_input_critic += self.critic_representation[self.model_keys[0]].output_shapes['state'][0]
        dim_input_critic += sum(self.n_actions.values())
        dim_input_critic += self.n_agents
        self.n_actions_max = max(self.n_actions.values())
        self.critic = BasicQhead(dim_input_critic, self.n_actions_max,
                                 critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic = deepcopy(self.critic)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ['RANK'])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_representation[key]._get_name() != "Basic_Identical":
                    self.critic_representation[key] = DistributedDataParallel(self.critic_representation[key],
                                                                              device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
            self.critic = DistributedDataParallel(module=self.critic, device_ids=[self.rank])

    @property
    def parameters_actor(self):
        return list(self.actor_representation.parameters()) + list(self.actor.parameters())

    @property
    def parameters_critic(self):
        return list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, epsilon=0.0, test_mode=False):
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
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            pi_logits[key] = self.actor[key](actor_input)
            act_probs[key] = nn.functional.softmax(pi_logits[key], dim=-1)

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
            critic_inputs.append(state.unsqueeze(-2).repeat(1, 1, self.n_agents, 1))  # batch * T * N * dim_S
        else:
            critic_inputs.append(state.unsqueeze(-2).repeat(1, self.n_agents, 1))  # batch * N * dim_S

        obs_rep = {}
        for key in self.model_keys:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]
            obs_rep[key] = outputs['state']

        agent_mask = (1 - torch.eye(self.n_agents, dtype=torch.float32, device=self.device)).unsqueeze(-1)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            agent_mask = agent_mask.repeat(1, 1, self.n_actions[key]).reshape(self.n_agents, -1).unsqueeze(0)
            if self.use_rnn:
                actions_input = actions[key].reshape(batch_size, seq_len, 1, -1).repeat(1, 1, self.n_agents, 1)
                critic_inputs.append(obs_rep[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
                critic_inputs.append(actions_input * agent_mask.unsqueeze(0))
                critic_inputs.append(agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2))
            else:
                actions_input = actions[key].reshape(batch_size, 1, -1).repeat(1, self.n_agents, 1)
                critic_inputs.append(obs_rep[key].reshape(batch_size, self.n_agents, -1))
                critic_inputs.append(actions_input * agent_mask)
                critic_inputs.append(agent_ids.reshape(batch_size, self.n_agents, -1))
            critic_inputs = torch.cat(critic_inputs, dim=-1)
        else:
            agent_mask = torch.cat([agent_mask[i].repeat(1, self.n_actions[k])
                                    for i, k in enumerate(self.model_keys)], dim=-1).unsqueeze(0)
            if self.use_rnn:
                agent_mask = agent_mask.unsqueeze(1)
                actions_input = torch.cat(itemgetter(*self.model_keys)(actions),
                                          dim=-1).unsqueeze(-2).repeat(1, 1, self.n_agents, 1)  # batch * T * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2)
            else:
                actions_input = torch.cat(itemgetter(*self.model_keys)(actions),
                                          dim=-1).unsqueeze(1).repeat(1, self.n_agents, 1)  # batch_size * N * A
                agent_ids = agent_ids.reshape(batch_size, self.n_agents, -1)  # batch_size * N * N
            critic_inputs.append(torch.stack(itemgetter(*self.model_keys)(obs_rep), dim=-2))
            critic_inputs.append(actions_input * agent_mask)
            critic_inputs.append(agent_ids)
            critic_inputs = torch.cat(critic_inputs, dim=-1)

        values = self.target_critic(critic_inputs) if target else self.critic(critic_inputs)
        return rnn_hidden_new, values

    def copy_target(self):
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(ep)


class MeanFieldActorCriticPolicy(Module):
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
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
                n_agents,)
            dim_critic_in += kwargs['action_embedding_hidden_size'][-1]

            self.action_mean_embedding[key] = Basic_MLP((dim_action_embedding,),
                                                        kwargs['action_embedding_hidden_size'],
                                                        normalize, initialize, activation, device)
            self.actor[key] = ActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                       normalize, initialize, activation, None, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = kwargs['temperature']

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_representation[key]._get_name() != "Basic_Identical":
                    self.critic_representation[key] = DistributedDataParallel(self.critic_representation[key],
                                                                              device_ids=[self.rank])
                self.action_mean_embedding[key] = DistributedDataParallel(module=self.action_mean_embedding[key],
                                                                          device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic[key] = DistributedDataParallel(module=self.critic[key], device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters()) + list(
            self.critic_representation.parameters()) + list(self.critic.parameters()) + list(
            self.action_mean_embedding.parameters())
        return parameters

    def get_boltzmann_policy(self, q):
        actions_prob = self.softmax(q / self.temperature)
        return actions_prob

    def get_mean_actions(self, actions: Dict[str, Tensor],
                         agent_mask_tensor: Tensor, batch_size: int):
        if self.use_parameter_sharing:
            actions_tensor = actions[self.model_keys[0]].reshape([-1, self.n_agents])
        else:
            actions_tensor = torch.stack(itemgetter(*self.model_keys)(actions), dim=-1).reshape([-1, self.n_agents])
        actions_onehot = one_hot(actions_tensor, num_classes=self.n_actions_max)

        # count alive neighbors
        _eyes = torch.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        agent_mask_diagonal = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) * _eyes
        agent_mask_neighbors = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) - agent_mask_diagonal
        agent_alive_neighbors = agent_mask_neighbors.sum(dim=-1, keepdim=True)

        # calculate mean actions of each agent's neighbors
        agent_mask_repeat = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_actions_max)
        actions_onehot = actions_onehot * agent_mask_repeat
        actions_sum = actions_onehot.sum(dim=-2, keepdim=True).repeat(1, self.n_agents, 1)
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

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
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
                actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_logits[key] = self.actor[key](actor_input, avail_actions_input)
            actions_prob = self.get_boltzmann_policy(pi_logits[key])
            pi_dists[key] = Categorical(probs=actions_prob)
        return rnn_hidden_new, pi_dists

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

        actions_mean = {key: Tensor(actions_mean[key]).to(self.device) for key in agent_list}
        for key in agent_list:
            if self.use_rnn:
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                action_embedding_input = torch.cat([actions_mean[key], agent_ids], dim=-1)
                act_embedding = self.action_mean_embedding[key](action_embedding_input)
                critic_input = torch.concat([outputs['state'], act_embedding['state'], agent_ids], dim=-1)
            else:
                act_embedding = self.action_mean_embedding[key](actions_mean[key])
                critic_input = torch.cat([outputs['state'], act_embedding['state']], dim=-1)

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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = actor_representation
        self.critic_1_representation = critic_representation
        self.critic_2_representation = deepcopy(critic_representation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)

        self.actor, self.critic_1, self.critic_2 = ModuleDict(), ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_1_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = Actor_SAC(dim_actor_in, dim_actor_out, actor_hidden_size,
                                        normalize, initialize, activation, device)
            self.critic_1[key] = BasicQhead(dim_critic_in, dim_action, critic_hidden_size,
                                            normalize, initialize, activation, device)
            self.critic_2[key] = BasicQhead(dim_critic_in, dim_action, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_1_representation[key]._get_name() != "Basic_Identical":
                    self.critic_1_representation[key] = DistributedDataParallel(self.critic_1_representation[key],
                                                                                device_ids=[self.rank])
                if self.critic_2_representation[key]._get_name() != "Basic_Identical":
                    self.critic_2_representation[key] = DistributedDataParallel(self.critic_2_representation[key],
                                                                                device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic_1[key] = DistributedDataParallel(module=self.critic_1[key], device_ids=[self.rank])
                self.critic_2[key] = DistributedDataParallel(module=self.critic_2[key], device_ids=[self.rank])

    @property
    def parameters_actor(self):
        parameters_actor = {}
        for key in self.model_keys:
            parameters_actor[key] = list(self.actor_representation[key].parameters()) + list(
                self.actor[key].parameters())
        return parameters_actor

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = list(self.critic_1_representation[key].parameters()) + list(
                self.critic_1[key].parameters()) + list(self.critic_2_representation[key].parameters()) + list(
                self.critic_2[key].parameters())
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor network and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor representation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic representation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden (Optional[Dict[str, List[Tensor]]]): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new (Optional[Dict[str, List[Tensor]]]): The new hidden variables of the RNN.
            actions (Dict[Tensor]): The actions output by the policies.
        """
        rnn_hidden_new, act_dists, actions_dict, log_action_prob = deepcopy(rnn_hidden), {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new.update({key: (outputs['rnn_hidden'], outputs['rnn_cell'])})
            else:
                outputs = self.actor_representation[key](observation[key])

            if self.use_parameter_sharing:
                actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_in = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            act_dists = self.actor[key](actor_in, avail_actions_input)
            actions_dict[key] = act_dists.stochastic_sample()
        return rnn_hidden_new, actions_dict, None

    def Qpolicy(self, observation: Dict[str, Tensor],
                agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None,
                agent_key: str = None,
                rnn_hidden_actor: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_actor (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for actor representation.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_actor_new: The updated rnn states for actor_representation.
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            act_prob_dict: The probabilities of actions.
            log_action_prob_dict: The log of action probabilities.
            q_1: The evaluation of Q values with critic 1.
            q_2: The evaluation of Q values with critic 2.
        """
        rnn_hidden_actor_new = deepcopy(rnn_hidden_actor)
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        act_prob_dict, log_action_prob_dict, q_1, q_2 = {}, {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs_actor = self.actor_representation[key](observation[key], *rnn_hidden_actor[key])
                outputs_critic_1 = self.critic_1_representation[key](observation[key], *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key], *rnn_hidden_critic_2[key])
                rnn_hidden_actor_new.update({key: (outputs_actor['rnn_hidden'], outputs_actor['rnn_cell'])})
                rnn_hidden_critic_new_1.update({key: (outputs_critic_1['rnn_hidden'], outputs_critic_1['rnn_cell'])})
                rnn_hidden_critic_new_2.update({key: (outputs_critic_2['rnn_hidden'], outputs_critic_2['rnn_cell'])})
            else:
                outputs_actor = self.actor_representation[key](observation[key])
                outputs_critic_1 = self.critic_1_representation[key](observation[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key])

            actor_in = outputs_actor['state']
            critic_1_in = outputs_critic_1['state']
            critic_2_in = outputs_critic_2['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([actor_in, agent_ids], dim=-1)
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            actor_dist = self.actor[key](actor_in, avail_actions_input)
            act_prob_dict[key] = actor_dist.probs
            z = act_prob_dict[key] <= 1e-20
            z = z.float() * 1e-8
            log_action_prob_dict[key] = torch.log(act_prob_dict[key] + z)

            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        return (rnn_hidden_actor_new, rnn_hidden_critic_new_1, rnn_hidden_critic_new_2,
                act_prob_dict, log_action_prob_dict, q_1, q_2)

    def Qtarget(self, next_observation: Dict[str, Tensor],
                agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None,
                agent_key: str = None,
                rnn_hidden_actor: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_actor (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for actor representation.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_actor: The updated rnn states for actor_representation.
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_actor_new = deepcopy(rnn_hidden_actor)
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        new_act_prob_dict, log_action_prob_dict, target_q = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs_actor = self.actor_representation[key](next_observation[key], *rnn_hidden_actor[key])
                outputs_critic_1 = self.target_critic_1_representation[key](next_observation[key],
                                                                            *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.target_critic_2_representation[key](next_observation[key],
                                                                            *rnn_hidden_critic_2[key])
                rnn_hidden_actor_new.update({key: (outputs_actor['rnn_hidden'], outputs_actor['rnn_cell'])})
                rnn_hidden_critic_new_1.update({key: (outputs_critic_1['rnn_hidden'], outputs_critic_1['rnn_cell'])})
                rnn_hidden_critic_new_2.update({key: (outputs_critic_2['rnn_hidden'], outputs_critic_2['rnn_cell'])})
            else:
                outputs_actor = self.actor_representation[key](next_observation[key])
                outputs_critic_1 = self.target_critic_1_representation[key](next_observation[key])
                outputs_critic_2 = self.target_critic_2_representation[key](next_observation[key])

            actor_in = outputs_actor['state']
            critic_1_in = outputs_critic_1['state']
            critic_2_in = outputs_critic_2['state']
            if self.use_parameter_sharing:
                actor_in = torch.concat([actor_in, agent_ids], dim=-1)
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            new_act_dist = self.actor[key](actor_in, avail_actions_input)
            new_act_prob_dict[key] = new_act_dist.probs
            z = new_act_prob_dict[key] <= 1e-20
            z = z.float() * 1e-8  # avoid log(0)
            log_action_prob_dict[key] = torch.log(new_act_prob_dict[key] + z)

            target_q_1, target_q_2 = self.target_critic_1[key](critic_1_in), self.target_critic_2[key](critic_2_in)
            target_q[key] = torch.min(target_q_1, target_q_2)
        return (rnn_hidden_actor_new, rnn_hidden_critic_new_1, rnn_hidden_critic_new_2,
                new_act_prob_dict, log_action_prob_dict, target_q)

    def Qaction(self, observation: Union[np.ndarray, dict],
                agent_ids: Tensor, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            observation (Union[np.ndarray, dict]): The original observation.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs_critic_1 = self.critic_1_representation[key](observation[key], *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key], *rnn_hidden_critic_2[key])
                rnn_hidden_critic_new_1.update({key: (outputs_critic_1['rnn_hidden'], outputs_critic_1['rnn_cell'])})
                rnn_hidden_critic_new_2.update({key: (outputs_critic_2['rnn_hidden'], outputs_critic_2['rnn_cell'])})
            else:
                outputs_critic_1 = self.critic_1_representation[key](observation[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key])

            critic_1_in = outputs_critic_1['state']
            critic_2_in = outputs_critic_2['state']
            if self.use_parameter_sharing:
                critic_1_in = torch.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = torch.concat([critic_2_in, agent_ids], dim=-1)
            q_1[key], q_2[key] = self.critic_1[key](critic_1_in), self.critic_2[key](critic_2_in)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


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
        device (Optional[Union[str, int, torch.device]]): The calculating device.
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
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, device,
                                           use_distributed_training, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor network and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor representation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic representation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self, joint_observation: Optional[Tensor] = None,
                joint_actions: Optional[Tensor] = None,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (Optional[Tensor]): The joint observations of the team.
            joint_actions (Optional[Tensor]): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The evaluations of Q^policy with critic 1.
            q_2: The evaluations of Q^policy with critic 2.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = torch.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = torch.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def Qtarget(self, joint_observation: Optional[Tensor] = None,
                joint_actions: Optional[Tensor] = None,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            joint_observation (Optional[Tensor]): The joint observations of the team.
            joint_actions (Optional[Tensor]): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = torch.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = torch.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1 = self.target_critic_1[key](critic_1_in)
            q_2 = self.target_critic_2[key](critic_2_in)
            target_q[key] = torch.min(q_1, q_2)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q

    def Qaction(self, joint_observation: Optional[Tensor] = None,
                joint_actions: Optional[Tensor] = None,
                agent_ids: Optional[Tensor] = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            joint_observation (Optional[Tensor]): The joint observations of the team.
            joint_actions (Tensor): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = torch.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = torch.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2

            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2
