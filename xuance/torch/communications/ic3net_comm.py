import torch
import torch.nn as nn
from numpy import ndarray

from xuance.common import Optional, Union, Sequence
from xuance.torch import Module


class IC3NetComm(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.device = device
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
        self.comm_passes = comm_passes

        self.obs_encoder = nn.Linear(input_shape[0], self.recurrent_hidden_size).to(self.device)
        self.msg_encoder = nn.ModuleList([nn.Linear(self.recurrent_hidden_size, self.recurrent_hidden_size).to(self.device)
                                        for _ in range(self.comm_passes)])
        self.tanh = nn.Tanh()
        # gate block
        self.gate = nn.Sequential(
            nn.Linear(self.recurrent_hidden_size, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Sigmoid(),
        ).to(self.device)

    def forward_state_encoder(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        observations = self.tanh(observations)
        return self.obs_encoder(observations)

    def build_actor_input(self, observations: torch.Tensor, msg: ndarray) -> torch.Tensor:
        msg = torch.as_tensor(msg, dtype=torch.float32, device=self.device)
        for i in range(self.comm_passes):
            msg = self.msg_encoder[i](msg)
            msg = self.tanh(msg)
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return observations + msg

    def get_message(self, hidden_state: torch.Tensor) -> torch.Tensor:
        prob = self.gate(hidden_state)
        msg = hidden_state * prob
        return msg