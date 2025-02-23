import torch
import torch.nn as nn
from xuance.common import Optional, Callable, Union, Sequence
from xuance.torch import Module, Tensor
from xuance.torch.utils import mlp_block, ModuleType


class Ic3NetComm(Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_size: int,
                 dim_actions: int,
                 n_action_heads: Sequence[int],
                 init_std: float = None,
                 recurrent: bool = True,
                 comm_init: str = "zeros",
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.dim_actions = dim_actions
        self.hid_size = hidden_size
        self.n_action_heads = n_action_heads
        self.init_std = init_std
        self.recurrent = recurrent
        self.device = device
        self.use_parameter_sharing = kwargs['use_parameter_sharing']

        if self.continuous:
            self.action_mean = nn.Linear(self.hid_size, self.dim_actions, device=self.device)
            self.action_log_std = nn.Parameter(torch.zeros(1, self.dim_actions)).to(self.device)
        else:
            self.heads = nn.ModuleList([nn.Linear(self.hid_size, o)
                                        for o in self.n_action_heads]).to(self.device)
        self.init_std = self.init_std if self.init_std is not None else 0.2
        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) - torch.eye(self.nagents, self.nagents)
        self.comm_mask = self.comm_mask.to(self.device)
        self.encoder = nn.Linear(self.state_dim, self.hid_size, device=self.device)
        if self.recurrent:
            self.hidd_encoder = nn.Linear(self.hid_size, self.hid_size, device=self.device)
        if self.use_parameter_sharing:
            self.C_module = nn.Linear(self.hid_size, self.hid_size, device=self.device)
            self.C_modules = nn.ModuleList([self.C_module for _ in range(self.comm_passes)]).to(self.device)
        else:
            self.C_modules = nn.ModuleList([nn.Linear(self.hid_size, self.hid_size)
                                            for _ in range(self.comm_passes)]).to(self.device)
        if self.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()
        self.value_head = nn.Linear(self.hidden_size, 1, device=self.device)

    def forward(self, hidden_features: Tensor):
        encoded_msg = self.msg_encoder(hidden_features)
        return encoded_msg


