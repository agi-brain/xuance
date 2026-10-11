from copy import deepcopy
from gymnasium.spaces import Space, Discrete
from typing import Type, Sequence, Optional, Callable, Union, Tuple

from xuance.mindspore import ms, ops, Tensor, Module, Parameter
from xuance.mindspore.rl_models.modules import ModelOutput
from xuance.mindspore.rl_models.heads import (QValueHead, DuelingQValueHead, C51QValueHead,
                                              QuantileRegressionQValueHead, RecurrentQValueHead)


class DeepQNetwork(Module):
    q_head_cls = QValueHead

    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space] = None,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
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
        )
        self.target_Q_head = deepcopy(self.eval_Q_head)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_embeddings, _ = self.representation(observation)
        q_values = self.eval_Q_head(rep_embeddings)
        greedy_actions = q_values.argmax(dim=-1)
        return q_values, greedy_actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> Tensor:
        target_rep_embeddings, _ = self.target_representation(observation)
        target_q_values = self.target_Q_head(target_rep_embeddings)
        return target_q_values

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Q_head.trainable_params(), self.target_Q_head.trainable_params()):
            tp.assign_value(ep)


class DuelingDeepQNetwork(DeepQNetwork):
    q_head_cls = DuelingQValueHead


class NoisyDeepQNetwork(DeepQNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = 0.0
        self.eval_noise_parameter = []
        self.target_noise_parameter = []

        self._stdnormal = ms.ops.StandardNormal()
        self._assign = ms.ops.Assign()

    def trainable_params(self, recurse=True):
        return self.representation.trainable_params() + self.eval_Q_head.trainable_params()

    def update_noise(self, noisy_bound: float = 0.0):
        """Updates the noises for network parameters."""
        self.eval_noise_parameter = []
        self.target_noise_parameter = []
        for parameter in self.eval_Q_head.trainable_params():
            self.eval_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)
            self.target_noise_parameter.append(self._stdnormal(parameter.shape) * noisy_bound)

    def update_parameters(self, is_target=False):
        self.update_noise(self.noise_scale)
        if is_target:
            for parameter, noise_param in zip(self.target_Q_head.trainable_params(), self.target_noise_parameter):
                parameter.assign_value(parameter.data + noise_param)
        else:
            for parameter, noise_param in zip(self.eval_Q_head.trainable_params(), self.eval_noise_parameter):
                parameter.assign_value(parameter.data + noise_param)


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
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
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
        )
        self.target_Z_head = deepcopy(self.eval_Z_head)
        self.supports = Parameter(
            ops.linspace(
                Tensor(self.v_min, dtype=ms.float32),
                Tensor(self.v_max, dtype=ms.float32),
                self.atom_num),
            requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atom_num - 1)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_embeddings, _ = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_embeddings)
        eval_Q = (self.supports * eval_Z).sum(-1)
        greedy_actions = eval_Q.argmax(dim=-1)
        return eval_Z, greedy_actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> Tuple[Tensor, ...]:
        target_rep_embeddings, _ = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_embeddings)
        target_Q = (self.supports * target_Z).sum(-1)
        argmax_action = target_Q.argmax(dim=-1)
        return target_Z, argmax_action

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Z_head.trainable_params(), self.target_Z_head.trainable_params()):
            tp.assign_value(ep)


class QRDeepQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 hidden_size: Sequence[int],
                 action_space: Optional[Space],
                 quantile_num: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
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
        )
        self.target_Z_head = deepcopy(self.eval_Z_head)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def construct(self,
                  observation: Union[Tensor, dict],
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_embeddings, _ = self.representation(observation)
        eval_Z = self.eval_Z_head(rep_embeddings)
        eval_Q = eval_Z.mean(dim=-1)
        greedy_actions = eval_Q.argmax(dim=-1)
        return eval_Z, greedy_actions

    def target(self,
               observation: Union[Tensor, dict],
               **kwargs) -> Tuple[Tensor, ...]:
        target_rep_embeddings, _ = self.target_representation(observation)
        target_Z = self.target_Z_head(target_rep_embeddings)
        target_Q = target_Z.mean(dim=-1)
        argmax_action = target_Q.argmax(dim=-1)
        return target_Z, argmax_action

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Z_head.trainable_params(), self.target_Z_head.trainable_params()):
            tp.assign_value(ep)


class DeepRecurrentQNetwork(Module):
    def __init__(self,
                 representation: Module,
                 recurrent_hidden_size: int,
                 recurrent_layer_N: int,
                 dropout: float,
                 action_space: Optional[Space] = None,
                 rnn: str = 'GRU',
                 initializer: Optional[Callable[..., Tensor]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError('action_space must be Discrete')
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
        )
        self.target_Q_head = deepcopy(self.eval_Q_head)

        self.lstm = self.eval_Q_head.lstm

        # Prepare DDP module.
        self.distributed_training = use_distributed_training

    def construct(self,
                  observation: Union[Tensor, dict],
                  rnn_states: Union[Tensor, Tuple[Tensor, Tensor]] = None,
                  **kwargs) -> Tuple[Tensor, ...]:
        rep_output, _ = self.representation(observation)
        rnn_states_new, q_values = self.eval_Q_head(rep_output, rnn_states)
        greedy_actions = q_values[:, -1].argmax(dim=-1)
        return rnn_states_new, q_values, greedy_actions

    def target(self,
               observation: Union[Tensor, dict],
               rnn_states: Union[Tensor, Tuple[Tensor, Tensor]] = None,
               **kwargs) -> Tuple[Tensor, ...]:
        target_rep_output, _ = self.target_representation(observation)
        rnn_states_new, target_q_values = self.target_Q_head(target_rep_output, rnn_states)
        argmax_action = target_q_values.argmax(dim=-1)
        return rnn_states_new, target_q_values, argmax_action

    def init_rnn_states(self, batch: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        hidden_states = ops.zeros(size=(self.recurrent_layer_N, batch, self.recurrent_hidden_size), dtype=ms.float32)
        if self.lstm:
            cell_states = ops.zeros_like(hidden_states)
            return hidden_states, cell_states
        else:
            return hidden_states

    def init_rnn_states_item(
            self,
            rnn_states: Union[Tensor, Tuple[Tensor, Tensor]],
            i: int
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.lstm:
            rnn_states[0][:, i] = ops.zeros(
                size=(self.recurrent_layer_N, self.recurrent_hidden_size), dtype=ms.float32)
            rnn_states[1][:, i] = ops.zeros(
                size=(self.recurrent_layer_N, self.recurrent_hidden_size), dtype=ms.float32)
            return rnn_states
        else:
            rnn_states[:, i] = ops.zeros(
                size=(self.recurrent_layer_N, self.recurrent_hidden_size), dtype=ms.float32)
        return rnn_states

    def copy_target(self):
        for ep, tp in zip(self.representation.trainable_params(), self.target_representation.trainable_params()):
            tp.assign_value(ep)
        for ep, tp in zip(self.eval_Q_head.trainable_params(), self.target_Q_head.trainable_params()):
            tp.assign_value(ep)
