import tensorflow as tf
from typing import Sequence, Optional, Tuple
from xuance.tensorflow import keras, Tensor, Module
from xuance.tensorflow.rl_models.modules.layers import mlp_block, gru_block, lstm_block, ModuleType
from xuance.tensorflow.rl_models.modules.outputs import RepresentationOutput, RNN_State


class Basic_RNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalizer: Optional[Module] = None,
                 initializer: Optional[keras.initializers.Initializer] = None,
                 activation: Optional[ModuleType] = None,
                 **kwargs):
        super(Basic_RNN, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalizer = normalizer
        self.initializer = initializer
        self.activation = activation
        self.kwargs = kwargs

        self.fc_hidden_sizes = kwargs["fc_hidden_sizes"]
        self.recurrent_hidden_size = kwargs["recurrent_hidden_size"]
        self.N_recurrent_layer = kwargs["N_recurrent_layers"]
        self.dropout = kwargs["dropout"]
        self.lstm = True if kwargs["rnn"] == "LSTM" else False

        self.output_shapes = {'state': (self.recurrent_hidden_size,)}

        self.mlp, self.rnn, output_dim = self._create_network()

        if self.normalizer is not None:
            self.use_normalizer = True
            self.input_norm = self.normalizer(input_shape)
            self.norm_rnn = self.normalizer(output_dim)
        else:
            self.use_normalizer = False

    def _create_network(self) -> Tuple[Module, Module, int]:
        layers = []
        input_shape = self.input_shape

        for h in self.fc_hidden_sizes:
            mlp_layer, input_shape = mlp_block(
                input_shape[0], h, self.normalizer, self.activation, self.initializer
            )
            layers.extend(mlp_layer)

        if self.lstm:
            rnn_layer, input_shape = lstm_block(
                input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer, self.dropout, self.initializer
            )

        else:
            rnn_layer, input_shape = gru_block(
                input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer, self.dropout, self.initializer)

        return keras.Sequential(layers), rnn_layer, input_shape

    def call(
            self, x: Tensor, rnn_states: RNN_State, **kwargs
    ) -> RepresentationOutput:

        tensor_x = tf.cast(x, dtype=tf.float32)

        if self.use_normalizer:
            tensor_x = self.input_norm(tensor_x)

        mlp_output = self.mlp(tensor_x)

        if self.lstm:
            output, hidden_states, cell_states = self._forward_lstm(mlp_output, rnn_states)

            if self.use_normalizer:
                output = self.norm_rnn(output)

            rnn_states_new = RNN_State(
                hidden_states=tf.stop_gradient(hidden_states),
                cell_states=tf.stop_gradient(cell_states),
            )
        else:
            output, hidden_states = self._forward_gru(mlp_output, rnn_states)

            if self.use_normalizer:
                output = self.norm_rnn(output)

            rnn_states_new = RNN_State(
                hidden_states=tf.stop_gradient(hidden_states)
            )

        return RepresentationOutput(
            embeddings=output,
            rnn_states=rnn_states_new
        )

    def _forward_gru(
            self,
            x: Tensor,
            rnn_states: RNN_State,
    ) -> Tuple[Tensor, Tensor]:
        """
        XuanCe state format:
            [N_layers, B, H]

        Keras GRU state format:
            [B, H]
        """

        hidden_states_new = []

        # Single-layer GRU.
        if self.N_recurrent_layer == 1:
            initial_state = rnn_states.hidden_states[0]

            output, hidden_state = self.rnn(x, initial_state=initial_state)

            hidden_states_new.append(hidden_state)

        # Stacked GRU.
        else:
            output = x

            for layer_index, rnn_layer in enumerate(self.rnn.layers):
                initial_state = rnn_states.hidden_states[layer_index]

                output, hidden_state = rnn_layer(output, initial_state=initial_state)

                hidden_states_new.append(hidden_state)

        hidden_states_new = tf.stack(hidden_states_new, axis=0)

        return output, hidden_states_new

    def _forward_lstm(self, x: Tensor, rnn_states: RNN_State) -> Tuple[Tensor, Tensor, Tensor]:
        """
        XuanCe state format:
            hidden_states: [N_layers, B, H]
            cell_states:   [N_layers, B, H]

        Keras LSTM state format:
            h: [B, H]
            c: [B, H]
        """
        hidden_states_new = []
        cell_states_new = []

        # Single-layer LSTM.
        if self.N_recurrent_layer == 1:
            initial_state = [rnn_states.hidden_states[0], rnn_states.cell_states[0]]

            output, hidden_state, cell_state = self.rnn(x, initial_state=initial_state)

            hidden_states_new.append(hidden_state)
            cell_states_new.append(cell_state)

        # Stacked LSTM.
        else:
            output = x

            for layer_index, rnn_layer in enumerate(self.rnn.layers):
                initial_state = [rnn_states.hidden_states[layer_index], rnn_states.cell_states[layer_index]]

                output, hidden_state, cell_state = rnn_layer(output, initial_state=initial_state)

                hidden_states_new.append(hidden_state)
                cell_states_new.append(cell_state)

        hidden_states_new = tf.stack(hidden_states_new, axis=0)

        cell_states_new = tf.stack(cell_states_new, axis=0)

        return output, hidden_states_new, cell_states_new

    def init_rnn_states(self, batch: int) -> RNN_State:
        hidden_states = tf.zeros(shape=(self.N_recurrent_layer, batch, self.recurrent_hidden_size),
                                 dtype=tf.float32)

        if self.lstm:
            cell_states = tf.zeros_like(hidden_states)
            return RNN_State(hidden_states=hidden_states, cell_states=cell_states)

        return RNN_State(hidden_states=hidden_states)

    def init_rnn_states_item(
            self,
            indexes: list,
            rnn_states: RNN_State,
    ) -> RNN_State:

        hidden_states = self._reset_state_items(rnn_states.hidden_states, indexes)

        if self.lstm:
            cell_states = self._reset_state_items(rnn_states.cell_states, indexes)

            return RNN_State(hidden_states=hidden_states, cell_states=cell_states)

        return RNN_State(hidden_states=hidden_states)

    @staticmethod
    def _reset_state_items(
            states: Tensor,
            indexes: list,
    ) -> Tensor:
        # states:
        # [N_layers, B, H]

        batch_size = tf.shape(states)[1]
        mask = tf.ones(shape=(batch_size,), dtype=states.dtype)
        indexes = tf.convert_to_tensor(indexes, dtype=tf.int32)
        updates = tf.zeros(shape=(tf.shape(indexes)[0],), dtype=states.dtype)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indexes, axis=-1), updates)
        mask = tf.reshape(mask, shape=(1, -1, 1))
        return states * mask

    def get_rnn_states_item(self, i: int, rnn_states: RNN_State) -> RNN_State:
        if self.lstm:
            return RNN_State(hidden_states=rnn_states.hidden_states[:, i],
                             cell_states=rnn_states.cell_states[:, i])
        else:
            return RNN_State(hidden_states=rnn_states.hidden_states[:, i])

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            input_shape=self.input_shape,
            hidden_sizes=self.hidden_sizes,
            normalizer=self.normalizer,
            initializer=self.initializer,
            activation=self.activation,
            **self.kwargs
        ))
        return config

