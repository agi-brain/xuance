import os
import torch
from copy import deepcopy
from gymnasium.spaces import Space, Discrete
from typing import Type, Sequence, Optional, Callable, Union, Tuple
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from xuance.torch.rl_models.modules import ModelOutput, RNN_State
from xuance.torch.rl_models.heads import (
    QValueHead, 
    DuelingQValueHead,
    C51QValueHead,
    QuantileRegressionQValueHead,
    RecurrentQValueHead
)


class DeepQNetwork(Module):
    q_head_cls = QValueHead

    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space] = None,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.device = device
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = representation.output_shapes

        self.eval_Q_head = self.q_head_cls(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device
        )
        self.target_Q_head = deepcopy(self.eval_Q_head)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Q_head = DistributedDataParallel(module=self.eval_Q_head, device_ids=[self.rank])

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        q_values = self.eval_Q_head(rep_output.embeddings)
        greedy_actions = q_values.argmax(dim=-1)
        return ModelOutput(
            actions=greedy_actions,
            values=q_values,
            rep_out=rep_output
        )

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = torch.randint(low=0, high=self.n_actions, size=greedy_actions.shape, device=self.device)
            random_mask = torch.rand(greedy_actions.shape, device=self.device) < epsilon_greedy
            actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_q_values = self.target_Q_head(target_rep_output.embeddings)
        return ModelOutput(values=target_q_values)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Q_head.parameters(), self.target_Q_head.parameters()):
            tp.data.copy_(ep)


class DuelingDeepQNetwork(DeepQNetwork):
    q_head_cls = DuelingQValueHead


class NoisyDeepQNetwork(DeepQNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = 0.0
        self.eval_noise_parameter = []
        self.target_noise_parameter = []

    def update_noise(self, noisy_bound: float = 0.0):
        """Updates the noises for network parameters."""
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Q_head.parameters():
            self.eval_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)
            self.target_noise_parameter.append(torch.randn_like(parameter) * noisy_bound)

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Q_head.parameters(), self.eval_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        return super().forward(observation, **kwargs)

    def act(self,
            observation: Union[Tensor, dict],
            **kwargs) -> Tensor:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.eval_Q_head.parameters(), self.eval_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        return super().act(observation=observation, deterministic=True)

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        self.update_noise(self.noise_scale)
        for parameter, noise_param in zip(self.target_Q_head.parameters(), self.target_noise_parameter):
            parameter.data.copy_(parameter.data + noise_param)
        return super().target(observation, **kwargs)


class C51DeepQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space],
                 atom_num: int,
                 v_min: float,
                 v_max: float,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.device = device
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = representation.output_shapes

        self.atom_num = atom_num
        self.v_min = v_min
        self.v_max = v_max

        self.eval_Z_head = C51QValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            atom_num=self.atom_num,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device
        )
        self.target_Z_head = deepcopy(self.eval_Z_head)
        self.supports = torch.nn.Parameter(torch.linspace(self.v_min, self.v_max, self.atom_num),
                                           requires_grad=False).to(device)
        self.delta_z = (v_max - v_min) / (atom_num - 1)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Z_head = DistributedDataParallel(module=self.eval_Z_head, device_ids=[self.rank])

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_output.embeddings)
        eval_Q = (self.supports * eval_Z).sum(-1)
        greedy_actions = eval_Q.argmax(dim=-1)
        return ModelOutput(
            actions=greedy_actions,
            values=eval_Z,
            rep_out=rep_output
        )

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = torch.randint(low=0, high=self.n_actions, size=greedy_actions.shape, device=self.device)
            random_mask = torch.rand(greedy_actions.shape, device=self.device) < epsilon_greedy
            actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_output.embeddings)
        target_Q = (self.supports * target_Z).sum(-1)
        argmax_action = target_Q.argmax(dim=-1)
        return ModelOutput(actions=argmax_action, values=target_Z)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Z_head.parameters(), self.target_Z_head.parameters()):
            tp.data.copy_(ep)


class QRDeepQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space],
                 quantile_num: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.device = device
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = representation.output_shapes

        self.quantile_num = quantile_num
        self.eval_Z_head = QuantileRegressionQValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            hidden_size=hidden_size,
            n_actions=self.n_actions,
            atom_num=self.quantile_num,
            normalizer=normalizer,
            initializer=initializer,
            activation=activation,
            device=device
        )
        self.target_Z_head = deepcopy(self.eval_Z_head)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Z_head = DistributedDataParallel(module=self.eval_Z_head, device_ids=[self.rank])

    def forward(self,
                observation: Union[Tensor, dict],
                **kwargs) -> ModelOutput:
        rep_output = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_output.embeddings)
        eval_Q = eval_Z.mean(dim=-1)
        greedy_actions = eval_Q.argmax(dim=-1)
        return ModelOutput(
            actions=greedy_actions,
            values=eval_Z,
            rep_out=rep_output
        )

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = torch.randint(low=0, high=self.n_actions, size=greedy_actions.shape, device=self.device)
            random_mask = torch.rand(greedy_actions.shape, device=self.device) < epsilon_greedy
            actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> ModelOutput:
        target_rep_output = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_output.embeddings)
        target_Q = target_Z.mean(dim=-1)
        argmax_action = target_Q.argmax(dim=-1)
        return ModelOutput(actions=argmax_action, values=target_Z)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Z_head.parameters(), self.target_Z_head.parameters()):
            tp.data.copy_(ep)


class DeepRecurrentQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 recurrent_hidden_size: int,
                 recurrent_layer_N: int,
                 dropout: float,
                 action_space: Optional[Space] = None,
                 rnn: str = 'GRU',
                 initializer: Optional[Callable[..., Tensor]] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
        self.device = device
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = representation.output_shapes

        self.recurrent_layer_N = recurrent_layer_N
        self.recurrent_hidden_size = recurrent_hidden_size

        self.eval_Q_head = RecurrentQValueHead(
            feature_dim=self.representation_info_shape['state'][0],
            recurrent_hidden_size=recurrent_hidden_size,
            recurrent_layer_N=recurrent_layer_N,
            dropout=dropout,
            n_actions=self.n_actions,
            rnn=rnn,
            initializer=initializer,
            device=device
        )
        self.target_Q_head = deepcopy(self.eval_Q_head)

        self.lstm = self.eval_Q_head.lstm

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Q_head = DistributedDataParallel(module=self.eval_Q_head, device_ids=[self.rank])

    def forward(self,
                observation: Union[Tensor, dict],
                rnn_states: RNN_State,
                **kwargs) -> Tuple[RNN_State, ModelOutput]:
        rep_output = self.representation(observation)
        rnn_states_new, q_values = self.eval_Q_head(rep_output.embeddings, rnn_states)
        greedy_actions = q_values[:, -1].argmax(dim=-1)
        return rnn_states_new, ModelOutput(actions=greedy_actions, values=q_values, rep_out=rep_output)

    def act(self,
            observation: Union[Tensor, dict],
            deterministic: bool = True,
            epsilon_greedy: float = 0.0,
            **kwargs) -> Tensor:
        greedy_actions = self(observation).actions

        if deterministic or epsilon_greedy <= 0.0:
            actions = greedy_actions
        else:
            random_actions = torch.randint(low=0, high=self.n_actions, size=greedy_actions.shape, device=self.device)
            random_mask = torch.rand(greedy_actions.shape, device=self.device) < epsilon_greedy
            actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def target(self,
               observation: Union[Tensor, dict],
               rnn_states: RNN_State,
               **kwargs) -> Tuple[RNN_State, ModelOutput]:
        target_rep_output = self.target_representation(observation)
        target_rnn_out, target_q_values = self.target_Q_head(target_rep_output.embeddings, rnn_states)
        argmax_action = target_q_values.argmax(dim=-1)
        return target_rnn_out, ModelOutput(actions=argmax_action, values=target_q_values)

    def init_rnn_states(self, batch: int) -> RNN_State:
        hidden_states = torch.zeros(size=(self.recurrent_layer_N, batch, self.recurrent_hidden_size)).to(self.device)
        cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
        return RNN_State(hidden_states=hidden_states, cell_states=cell_states)

    def init_rnn_states_item(self, rnn_states: RNN_State, i: int) -> RNN_State:
        rnn_states.hidden_states[:, i] = torch.zeros(
            size=(self.recurrent_layer_N, self.recurrent_hidden_size)).to(self.device)
        if self.lstm:
            rnn_states.cell_states[:, i] = torch.zeros(
                size=(self.recurrent_layer_N, self.recurrent_hidden_size)).to(self.device)
            return rnn_states
        return rnn_states

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Q_head.parameters(), self.target_Q_head.parameters()):
            tp.data.copy_(ep)


