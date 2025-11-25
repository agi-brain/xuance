from copy import deepcopy
from operator import itemgetter
from types import ModuleType
from typing import Optional, Dict, Sequence, Callable, Union, List

import torch
from gymnasium.spaces import Discrete
from torch import Tensor, nn
from torch.nn import ModuleDict, Module
from xuance.torch.policies import CommNet_Policy



class DGN_Policy(CommNet_Policy):
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
        super(DGN_Policy, self).__init__(action_space, n_agents, representation_actor, representation_critic, mixer,
                                         actor_hidden_size, critic_hidden_size, normalize, initialize, activation,
                                         device, use_distributed_training, **kwargs)

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, alive_ally: Optional[dict] = None):
        rnn_hidden_new, pi_dists = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        seq_length = observation[self.model_keys[0]].shape[1]
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