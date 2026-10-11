from typing import Type, Optional, Sequence, Tuple, Callable, Union
from xuance.mindspore import nn, ops, Tensor, Module
from xuance.mindspore.rl_models.modules import mlp_block, gru_block, lstm_block
from xuance.mindspore.rl_models.modules.outputs import RNN_State


class QValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions, None, None, initializer)[0])
        self.q_value = nn.SequentialCell(*layers)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs) -> Tensor:
        q_values = self.q_value(features)
        if avail_actions is not None:
            q_values[avail_actions == 0] = -1e10
        return q_values


class DuelingQValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        v_layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h // 2, normalizer, activation, initializer)
            v_layers.extend(mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, normalizer)[0])
        self.v_model = nn.SequentialCell(*v_layers)

        a_layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalizer, activation, initializer)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], n_actions, None, None, normalizer)[0])
        self.a_model = nn.SequentialCell(*a_layers)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs) -> Tensor:
        values = self.v_model(features)
        advantages = self.a_model(features)
        q_values = values + (advantages - advantages.mean(dim=-1).unsqueeze(dim=-1))
        if avail_actions is not None:
            q_values[avail_actions == 0] = -1e10
        return q_values


class C51QValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 atom_num: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], self.n_actions * self.atom_num, None, None, initializer)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs) -> Tensor:
        dist_logits = self.model(features).view(-1, self.n_actions, self.atom_num)
        if avail_actions is not None:
            dist_logits[avail_actions == 0] = -1e10
        dist_probs = ops.softmax(dist_logits, axis=-1)
        return dist_probs


class QuantileRegressionQValueHead(C51QValueHead):
    def construct(self,
                  features: Tensor,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs) -> Tensor:
        quantiles = self.model(features).view(-1, self.n_actions, self.atom_num)
        return quantiles


class RecurrentQValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 recurrent_hidden_size: int,
                 recurrent_layer_N: int,
                 dropout: float,
                 n_actions: int,
                 rnn: str = 'GRU',
                 initializer: Optional[Callable[..., Tensor]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        if rnn == "GRU":
            self.lstm = False
            rnn_block = gru_block
        elif rnn == "LSTM":
            self.lstm = True
            rnn_block = lstm_block
        else:
            raise ValueError("Unknown recurrent module!")
        self.rnn_layer, _ = rnn_block(
            input_dim=self.feature_dim,
            output_dim=recurrent_hidden_size,
            num_layers=recurrent_layer_N,
            dropout=dropout,
            initialize=initializer,
        )

        fc_layer = mlp_block(recurrent_hidden_size, self.n_actions, None, None, None)[0]
        self.q_value = nn.SequentialCell(*fc_layer)

    def construct(self,
                  features: Tensor,
                  rnn_states: Union[Tensor, Tuple[Tensor, Tensor]] = None,
                  avail_actions: Optional[Tensor] = None,
                  **kwargs) -> Tuple[RNN_State, Tensor]:
        embeddings, hn = self.rnn_layer(features, rnn_states)
        rnn_output = hn

        q_values = self.q_value(embeddings)
        if avail_actions is not None:
            q_values[avail_actions == 0] = -1e10
        return rnn_output, q_values
