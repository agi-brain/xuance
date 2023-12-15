from optparse import Option
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow_addons as tfa
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union, Callable

ModelType = Type[tk.Model]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[tk.layers.Layer] = None,
              activation: Optional[tk.layers.Layer] = None,
              initializer: Optional[tk.initializers.Initializer] = None,
              device: str = "cpu:0"):
    with tf.device(device):
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
              initializer: Optional[tk.initializers.Initializer] = None,
              device: str = "cpu:0"):
    assert len(input_shape) == 3
    H, W, C = input_shape
    with tf.device(device):
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
                  pooling: Optional[tk.layers.Layer] = None,
                  device: str = "cpu") -> Sequence[ModelType]:
    assert len(input_shape) == 3  # CxHxW
    block = []
    C, H, W = input_shape
    block.append(pooling(output_size=(H // scale, W // scale), device=device))
    return block


def gru_block(input_dim: Sequence[int],
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
              device: str = "cpu") -> ModelType:
    gru = tk.layers.GRU(units=output_dim,
                        dropout=dropout,
                        return_sequences=True,
                        return_state=True)
    return gru, output_dim


def lstm_block(input_dim: Sequence[int],
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
               device: str = "cpu") -> ModelType:
    lstm = tk.layers.LSTM(units=output_dim,
                          dropout=dropout,
                          return_sequences=True,
                          return_state=True)
    return lstm, output_dim
