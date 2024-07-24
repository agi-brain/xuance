from xuance.common import Optional, Sequence, Type, Callable
from xuance.tensorflow import Module, Tensor, tk

ModuleType = Type[Module]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[tk.layers.Layer] = None,
              activation: Optional[tk.layers.Layer] = None,
              initializer: Optional[tk.initializers.Initializer] = None):
    block = []
    if initializer is not None:
        linear = tk.layers.Dense(units=output_dim,
                                 activation=activation,
                                 kernel_initializer=initializer,
                                 input_shape=(input_dim,))
    else:
        linear = tk.layers.Dense(units=output_dim,
                                 activation=activation,
                                 input_shape=(input_dim,))
    block.append(linear)
    if normalize is not None:
        block.append(normalize())
    return block, (output_dim,)


def cnn_block(input_shape: Sequence[int],
              filters: int,
              kernel_size: int,
              stride: int,
              normalize: Optional[tk.layers.Layer] = None,
              activation: Optional[tk.layers.Layer] = None,
              initializer: Optional[tk.initializers.Initializer] = None):
    assert len(input_shape) == 3
    H, W, C = input_shape
    block = []
    if initializer is not None:
        cnn = tk.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               strides=(stride, stride),
                               activation=activation,
                               kernel_initializer=initializer,
                               input_shape=input_shape)
    else:
        cnn = tk.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               strides=(stride, stride),
                               activation=activation,
                               input_shape=input_shape)
    block.append(cnn)
    if normalize is not None:
        block.append(normalize())

    if H % stride == 0:
        H = H // stride
    else:
        H = (H + stride) // stride
    if W % stride == 0:
        W = W // stride
    else:
        W = (W + stride) // stride
    return block, (H, W, filters)


def pooling_block(input_shape: Sequence[int],
                  scale: int,
                  pooling: Optional[tk.layers.Layer] = None) -> Sequence[ModuleType]:
    assert len(input_shape) == 3  # CxHxW
    block = []
    C, H, W = input_shape
    block.append(pooling(output_size=(H // scale, W // scale)))
    return block


def gru_block(input_dim: Sequence[int],
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[Tensor], Tensor]] = None):
    gru = tk.layers.GRU(units=output_dim,
                        dropout=dropout,
                        return_sequences=True,
                        return_state=True)
    return gru, output_dim


def lstm_block(input_dim: Sequence[int],
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[Tensor], Tensor]] = None):
    lstm = tk.layers.LSTM(units=output_dim,
                          dropout=dropout,
                          return_sequences=True,
                          return_state=True)
    return lstm, output_dim
