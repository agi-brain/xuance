from argparse import Namespace

import torch
import torch.nn as nn

from xuance.common import Optional, Union, Sequence
from xuance.torch import Module


class CommNet(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 comm_passes: Optional[int] = 1,
                 model_keys: dict = None,
                 agent_keys: dict = None,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None,
                 config: Optional[Namespace] = None,
                 **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.device = device
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
        self.comm_passes = comm_passes
        self.model_keys = model_keys
        self.agent_keys = agent_keys
        self.n_agents = n_agents

        self.config = config
        self.use_parameter_sharing = self.config.use_parameter_sharing
        self.obs_encoder = self.create_mlp(input_shape[0], [], self.recurrent_hidden_size, nn.LeakyReLU(), self.device)
        self.msg_encoder = self.create_mlp(self.recurrent_hidden_size, [], self.recurrent_hidden_size, nn.LeakyReLU(), self.device)

    def create_mlp(self, input_shape: int,
                    layers: list,
                    out_shape: int,
                    activation: nn.Module,
                    device: Union[str, torch.device]) -> nn.Sequential:

        network_layers = []
        if len(layers):
            network_layers.append(nn.Linear(input_shape, layers[0]))
            network_layers.append(activation)
            for i in range(len(layers) - 1):
                network_layers.append(nn.Linear(layers[i], layers[i + 1]))
                network_layers.append(activation)
            network_layers.append(nn.Linear(layers[-1], out_shape))
        else:
            network_layers = [nn.Linear(input_shape, out_shape)]

        return nn.Sequential(*network_layers).to(device=device)

    def message_encode(self, message: torch.Tensor) -> torch.Tensor:
        return self.msg_encoder(message)

    def obs_encode(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.obs_encoder(obs)

    def forward(self, obs: torch.Tensor, msg_send: dict, alive_ally: dict) -> torch.Tensor:
        alive_ally = {k: torch.as_tensor(alive_ally[k], dtype=torch.float32, device=self.device) for k in
                      self.agent_keys}
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            msg_send = msg_send[key].view(batch_size // self.n_agents, self.n_agents, seq_length, -1)
            alive_ally = torch.stack(list(alive_ally.values()), dim=1)
            alive_agent_num = torch.sum(alive_ally, dim=1).unsqueeze(1)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
            msg_send = msg_send * alive_ally
            message = torch.sum(msg_send, dim=1, keepdim=True) - msg_send
        else:
            message = {k: msg_send[k] * alive_ally[k] for k in self.agent_keys}
            message = torch.stack(list(message.values()), dim=1)
            alive_ally = torch.stack(list(alive_ally.values()), dim=1)
            alive_agent_num = torch.sum(alive_ally, dim=1)
            alive_agent_num = torch.clamp(alive_agent_num, min=1.0)
            message = torch.sum(message, dim=1)
        message = message / alive_agent_num
        msg_receive = self.message_encode(message)
        if self.use_parameter_sharing:
            msg_receive = msg_receive.view(batch_size, seq_length, -1)
        return msg_receive