from typing import Type, Optional, Sequence, Tuple
from xuance.tensorflow import tf, keras, Tensor, Module, ModuleType
from xuance.tensorflow.rl_models.modules import mlp_block, gru_block, lstm_block
from xuance.tensorflow.rl_models.modules.outputs import RNN_State


class QValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 normalizer: Optional[ModuleType] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions, None, None, initializer)[0])
        self.q_value = keras.Sequential(layers)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> Tensor:
        q_values = self.q_value(features)
        if avail_actions is not None:
            q_values = tf.where(tf.equal(avail_actions, 0), tf.cast(-1e10, q_values.dtype), q_values)
        return q_values

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            n_actions=self.n_actions,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class DuelingQValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        v_layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h // 2, normalizer, activation, initializer)
            v_layers.extend(mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, normalizer)[0])
        self.v_model = keras.Sequential(v_layers)

        a_layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalizer, activation, initializer)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], n_actions, None, None, normalizer)[0])
        self.a_model = keras.Sequential(a_layers)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> Tensor:
        values = self.v_model(features)
        advantages = self.a_model(features)
        q_values = values + (advantages - tf.reduce_mean(advantages, axis=-1, keepdims=True))
        if avail_actions is not None:
            q_values = tf.where(tf.equal(avail_actions, 0), tf.cast(-1e10, q_values.dtype), q_values)
        return q_values

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            n_actions=self.n_actions,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class C51QValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_size: Sequence[int],
                 n_actions: int,
                 atom_num: int,
                 normalizer: Optional[Type[Module]] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[Type[Module]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.atom_num = atom_num
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation

        layers = []
        input_shape = (feature_dim,)
        for h in hidden_size:
            mlp, input_shape = mlp_block(input_shape[0], h, normalizer, activation, initializer)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], self.n_actions * self.atom_num, None, None, initializer)[0])
        self.model = keras.Sequential(layers)

    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> Tensor:
        logits = tf.reshape(self.model(features), [-1, self.n_actions, self.atom_num])
        if avail_actions is not None:
            logits = tf.where(tf.equal(avail_actions, 0), tf.cast(-1e10, logits.dtype), logits)
        dist_probs = tf.nn.softmax(logits, axis=-1)
        return dist_probs

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            n_actions=self.n_actions,
            atom_num=self.atom_num,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
        ))
        return config


class QuantileRegressionQValueHead(C51QValueHead):
    def call(self,
             features: Tensor,
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> Tensor:
        quantiles = self.model(features)
        quantiles = tf.reshape(quantiles, [-1, self.n_actions, self.atom_num])
        return quantiles


class RecurrentQValueHead(Module):
    def __init__(self,
                 feature_dim: int,
                 recurrent_hidden_size: int,
                 recurrent_layer_N: int,
                 dropout: float,
                 n_actions: int,
                 rnn: str = 'GRU',
                 initializer: Optional[keras.initializers.Initializer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_layer_N = recurrent_layer_N
        self.dropout = dropout
        self.n_actions = n_actions
        self.rnn = rnn
        self.initializer = initializer

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
            initializer=initializer,
        )

        fc_layer = mlp_block(recurrent_hidden_size, self.n_actions, None, None, None)[0]
        self.q_value = keras.Sequential(fc_layer)

    def call(self,
             features: Tensor,
             rnn_states: RNN_State,
             avail_actions: Optional[Tensor] = None,
             **kwargs) -> Tuple[RNN_State, Tensor]:
        hidden_states_new = []
        cell_states_new = []

        x = features
        if self.lstm:
            hidden_states = rnn_states.hidden_states
            cell_states = rnn_states.cell_states
            for i, rnn_layer in enumerate(self.rnn_layer):
                x, h, c = rnn_layer(x, initial_state=[hidden_states[i], cell_states[i]])
                hidden_states_new.append(h)
                cell_states_new.append(c)

            hidden_states = tf.stack(hidden_states_new, axis=0)
            cell_states = tf.stack(cell_states_new, axis=0)
            embeddings = x
            rnn_output = RNN_State(hidden_states=hidden_states, cell_states=cell_states)

        else:
            hidden_states = rnn_states.hidden_states
            for i, rnn_layer in enumerate(self.rnn_layer):
                x, h = rnn_layer(x, initial_state=hidden_states[i])
                hidden_states_new.append(h)

            hidden_states = tf.stack(hidden_states_new, axis=0)
            embeddings = x
            rnn_output = RNN_State(hidden_states=hidden_states)

        q_values = self.q_value(embeddings)
        if avail_actions is not None:
            q_values = tf.where(tf.equal(avail_actions, 0), tf.cast(-1e10, q_values.dtype), q_values)
        return rnn_output, q_values

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            feature_dim=self.feature_dim,
            recurrent_hidden_size=self.recurrent_hidden_size,
            recurrent_layer_N=self.recurrent_layer_N,
            dropout=self.dropout,
            n_actions=self.n_actions,
            rnn=self.rnn,
            initializer=self.initializer
        ))
        return config

