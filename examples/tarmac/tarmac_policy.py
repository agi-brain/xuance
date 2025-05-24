import os
from typing import Optional, Dict, Sequence, Callable, Union, List

import torch
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.nn.parallel import DistributedDataParallel

from xuance.torch.policies import CategoricalActorNet, CriticNet
from xuance.torch.utils import ModuleType


class TarMAC_Policy(Module):

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 communicator: ModuleDict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(TarMAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic
        self.communicator = communicator

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
            self.critic_representation.parameters()) + list(self.critic.parameters()) + list(self.communicator.parameters())
        if self.mixer is not None:
            parameters += list(self.mixer.parameters())
        return parameters

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        # calculate message_receive
        message_receive = {k: self.communicator[k](k, observation[k], rnn_hidden, alive_ally) for k in
                               self.model_keys}

        for key in agent_list:
            outputs = self.actor_representation[key](message_receive[key], *rnn_hidden[key])
            rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            actor_input = outputs['state']
            avail_actions_input = None if avail_actions is None else avail_actions[key]
            pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
        return rnn_hidden_new, pi_dists

    def get_values(self, observation: Dict[str, Tensor], agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, values = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        # calculate message_receive
        message_receive = {k: self.communicator[k](self.model_keys, observation[k], rnn_hidden, alive_ally) for k in self.model_keys}
        message = {k: torch.cat(list(message_receive.values()), dim=-1) for k in self.model_keys}

        for key in agent_list:
            outputs = self.critic_representation[key](message[key], *rnn_hidden[key])
            rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            critic_input = outputs['state']
            values[key] = self.critic[key](critic_input)

        return rnn_hidden_new, values

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)