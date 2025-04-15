import torch
import torch.nn as nn
from gymnasium.spaces import Space
from xuance.common import Dict, Optional, Union, Sequence, space2shape
from xuance.torch import Module, Tensor, ModuleDict


class IC3NetComm(Module):
    def __init__(self,
                 obs_space: Union[Dict[str, Space], tuple],
                 act_space: Union[Dict[str, Space], tuple],
                 obs_encode_dim: int,
                 n_agents: int,
                 hidden_size: int,
                 n_action_heads: Sequence[int],
                 comm_mask_zero: bool = False,
                 comm_passes: int = 1,
                 init_std: float = 0.2,
                 recurrent: bool = True,
                 comm_init: str = 'zeros',
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.obs_encode_dim = obs_encode_dim
        self.n_agents = n_agents
        self.hid_size = hidden_size
        self.n_action_heads = n_action_heads
        self.comm_mask_zero = comm_mask_zero
        self.comm_passes = comm_passes
        self.comm_init = comm_init
        self.init_std = init_std
        self.recurrent = recurrent
        self.device = device
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.continuous = kwargs['continuous']

        self.encoder, self.hidd_encoder = ModuleDict(), ModuleDict()
        self.heads, self.value_head = ModuleDict(), ModuleDict()
        self.action_mean, self.action_log_std = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            self.encoder[key] = nn.Linear(space2shape(obs_space[key])[0], self.obs_encode_dim, device=device)
            if self.continuous:
                self.action_mean[key] = nn.Linear(self.hid_size, space2shape(act_space[key])[0], device=device)
                self.action_log_std[key] = nn.Parameter(torch.zeros(1, space2shape(act_space[key])[0])).to(self.device)
            else:
                self.heads[key] = nn.ModuleList([nn.Linear(self.hid_size, act_space[key].n),
                                                 nn.Linear(self.hid_size, 2)]).to(self.device)

            self.value_head[key] = nn.Linear(self.hid_size, 1, device=device)
            if self.recurrent:
                self.hidd_encoder[key] = nn.Linear(self.hid_size, self.hid_size, device=device)
        self.C_modules = nn.ModuleList([nn.Linear(self.hid_size, self.hid_size)
                                        for _ in range(self.comm_passes)]).to(self.device)  # share parameters.

        # Mask for communication
        if self.comm_mask_zero:
            self.comm_mask = torch.zeros(self.n_agents, self.n_agents)
        else:
            self.comm_mask = torch.ones(self.n_agents, self.n_agents) - torch.eye(self.n_agents, self.n_agents)
        self.comm_mask = self.comm_mask.to(self.device)

        if self.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

    def forward_state_encoder(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        obs_encode = {}
        for key in self.model_keys:
            x = Tensor(observations[key]).to(self.device)
            x = self.encoder[key](x)
            if self.recurrent:
                obs_encode[key] = x
            else:
                obs_encode[key] = self.tanh(x)
        return obs_encode

    def forward(self, hidden_features: Tensor):
        encoded_msg = self.msg_encoder(hidden_features)
        return encoded_msg
