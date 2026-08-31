import numpy as np
from typing import Optional, Sequence, Type, Callable, Tuple, List
from xuance.tensorflow import tf, keras, Tensor, Module

ModuleType = Type[Module]


def _init_layer(layer,
                gain: float = np.sqrt(2.0),
                bias: float = 0.0):
    """initializer a built Keras layer with orthogonal weights and constant bias."""
    if not layer.built:
        raise ValueError("The layer must be built before calling _init_layer().")

    if hasattr(layer, "kernel") and layer.kernel is not None:
        kernel = keras.initializers.Orthogonal(gain=gain)(
            shape=layer.kernel.shape,
            dtype=layer.kernel.dtype,
        )
        layer.kernel.assign(kernel)

    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.assign(tf.fill(layer.bias.shape, tf.cast(bias, layer.bias.dtype)))

    return layer


def mlp_block(
        input_dim: int,
        output_dim: int,
        normalizer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
        initializer: Optional[keras.initializers.Initializer] = None
) -> Tuple[keras.layers.Layer, int]:
    input_dim = int(input_dim)
    output_dim = int(output_dim)

    dense_kwargs = {
        "units": output_dim,
        "bias_initializer": "zeros",
    }

    if initializer is not None:
        dense_kwargs["kernel_initializer"] = keras.initializers.get(initializer)

    dense = keras.layers.Dense(**dense_kwargs)

    dense.build((None, input_dim))

    block = [dense]

    if activation is not None:
        block.append(keras.layers.Activation(keras.activations.get(activation)))

    if normalizer is not None:
        block.append(normalizer())

    return block, (output_dim,)


def cnn_block(
        input_shape: Sequence[int],
        filters: int,
        kernel_size: int,
        stride: int,
        normalizer: Optional[keras.layers.Layer] = None,
        activation: Optional[keras.layers.Layer] = None,
        initializer: Optional[keras.initializers.Initializer] = None
) -> Tuple[keras.layers.Layer, tuple]:
    assert len(input_shape) == 3
    H, W, C = input_shape

    padding = int((kernel_size - stride) // 2)
    block = []

    if padding > 0:
        block.append(keras.layers.ZeroPadding2D(padding=padding))
        H += 2 * padding
        W += 2 * padding

    cnn = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="valid"
    )
    cnn.build((None, H, W, C))

    if initializer is not None:
        initializerd_kernel = initializer(cnn.kernel)
        if initializerd_kernel is not None:
            cnn.kernel.assign(initializerd_kernel)
        cnn.bias.assign(tf.zeros_like(cnn.bias))

    block.append(cnn)

    C = filters
    H = int((H - kernel_size) / stride + 1)
    W = int((W - kernel_size) / stride + 1)

    if activation is not None:
        block.append(keras.layers.Activation(activation))

    if normalizer is not None:
        if normalizer == keras.layers.GroupNormalization:
            block.append(normalizer(groups=max(C // 2, 1), axis=-1))
        elif normalizer == keras.layers.LayerNormalization:
            block.append(normalizer(axis=[1, 2, 3]))
        else:
            block.append(normalizer(axis=-1))

    return block, (H, W, C)


def pooling_block(
        input_shape: Sequence[int],
        scale: int,
        pooling: Optional[keras.layers.Layer] = None
) -> Sequence[ModuleType]:
    assert len(input_shape) == 3  # H x W x C

    block = []

    block.append(
        pooling(
            pool_size=scale,
            strides=scale,
            padding="valid",
        )
    )
    return block


def gru_block(
        input_dim: Sequence[int],
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        initializer: Optional[Callable[[Tensor], Tensor]] = None
) -> Tuple[List[keras.layers.Layer], int]:
    layers = []

    for layer_index in range(num_layers):
        dense_kwargs = {
            "units": output_dim,
            "dropout": dropout if num_layers > 1 else 0.0,
            "return_sequences": True,
            "return_state": True
        }

        current_input_dim = input_dim if layer_index == 0 else output_dim

        if initializer is not None:
            dense_kwargs["kernel_initializer"] = keras.initializers.get(initializer)

        gru = keras.layers.GRU(**dense_kwargs)
        gru.build((None, None, current_input_dim))

        layers.append(gru)

    return layers, output_dim


def lstm_block(
        input_dim: Sequence[int],
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        initializer: Optional[Callable[[Tensor], Tensor]] = None
) -> Tuple[List[keras.layers.Layer], int]:
    layers = []

    for layer_index in range(num_layers):
        dense_kwargs = {
            "units": output_dim,
            "dropout": dropout,
            "return_sequences": True,
            "return_state": True
        }

        if initializer is not None:
            dense_kwargs["kernel_initializer"] = keras.initializers.get(initializer)

        lstm = keras.layers.LSTM(**dense_kwargs)
        current_input_dim = input_dim if layer_index == 0 else output_dim
        lstm.build((None, None, current_input_dim))

        layers.append(lstm)

    return layers, output_dim
