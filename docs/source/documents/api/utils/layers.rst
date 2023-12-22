Neural Network Layers
=======================================

This module defines utility functions to create blocks commonly used in neural network architectures.

.. raw:: html

    <br><hr>

PyTorch
---------------------------------------------------


.. py:function::
  xuance.torch.utils.layers.mlp_block(input_dim, output_dim, normalize, activation, initialize, device)

  This function creates a block for a multi-layer perceptron (MLP) or fully connected layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing a sequence of modules representing the MLP block and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.torch.utils.layers.cnn_block(input_shape, filter, kernel_size, stride, normalize, activation, initialize, device)

  This function creates a block for a convolutional neural network (CNN) layer.

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param filter: Number of filters.
  :type filter: int
  :param kernel_size: Size of the convolutional kernel.
  :type kernel_size: int
  :param stride: Stride of the convolution.
  :type stride: int
  :param normalize: The method of normalization.
  :type normalize: nn.Module
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Module
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing a sequence of modules representing the CNN block and the updated output shape (C, H, W).
  :rtype: tuple

.. py:function::
  xuance.torch.utils.layers.pooling_block(input_shape, scale, pooling, device)

  This function creates a block for pooling (either AdaptiveMaxPool2d or AdaptiveAvgPool2d).

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param scale: Scaling factor for pooling.
  :type scale: float
  :param pooling: Pooling layer (either AdaptiveMaxPool2d or AdaptiveAvgPool2d).
  :type pooling: int
  :param device: The calculating device.
  :type device: str
  :return: A sequence of modules representing the pooling block.
  :rtype: list

.. py:function::
  xuance.torch.utils.layers.gru_block(input_dim, output_dim, num_layers, dropout, initialize)

  This function creates a block for a Gated Recurrent Unit (GRU) layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of GRU layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :return: A tuple containing the GRU module and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.torch.utils.layers.lstm_block(input_dim, output_dim, num_layers, dropout, initialize, device)

  This function creates a block for a Long Short-Term Memory (LSTM) layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of LSTM layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing the LSTM module and the output dimension.
  :rtype: tuple

.. raw:: html

    <br><hr>

TensorFlow
--------------------------------

.. py:function::
  xuance.tensorflow.utils.layers.mlp_block(input_dim, output_dim, normalize, activation, initialize, device)

  This function creates a block for a multi-layer perceptron (MLP) or fully connected layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing a sequence of modules representing the MLP block and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.utils.layers.cnn_block(input_shape, filter, kernel_size, stride, normalize, activation, initialize, device)

  This function creates a block for a convolutional neural network (CNN) layer.

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param filter: Number of filters.
  :type filter: int
  :param kernel_size: Size of the convolutional kernel.
  :type kernel_size: int
  :param stride: Stride of the convolution.
  :type stride: int
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing a sequence of modules representing the CNN block and the updated output shape (C, H, W).
  :rtype: tuple

.. py:function::
  xuance.tensorflow.utils.layers.pooling_block(input_shape, scale, pooling, device)

  This function creates a block for pooling (either AdaptiveMaxPool2d or AdaptiveAvgPool2d).

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param scale: Scaling factor for pooling.
  :type scale: float
  :param pooling: Pooling layer (either AdaptiveMaxPool2d or AdaptiveAvgPool2d).
  :type pooling: int
  :param device: The calculating device.
  :type device: str
  :return: A sequence of modules representing the pooling block.
  :rtype: list

.. py:function::
  xuance.tensorflow.utils.layers.gru_block(input_dim, output_dim, num_layers, dropout, initialize)

  This function creates a block for a Gated Recurrent Unit (GRU) layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of GRU layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :return: A tuple containing the GRU module and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.utils.layers.lstm_block(input_dim, output_dim, num_layers, dropout, initialize, device)

  This function creates a block for a Long Short-Term Memory (LSTM) layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of LSTM layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param device: The calculating device.
  :type device: str
  :return: A tuple containing the LSTM module and the output dimension.
  :rtype: tuple

.. raw:: html

    <br><hr>

MindSpore
------------------------------------

.. py:function::
  xuance.mindspore.utils.layers.mlp_block(input_dim, output_dim, normalize, activation, initialize)

  This function creates a block for a multi-layer perceptron (MLP) or fully connected layer.

  :param input_dim: the dimension of the input data.
  :type input_dim: int
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :return: A tuple containing a sequence of modules representing the MLP block and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.mindspore.utils.layers.cnn_block(input_shape, filter, kernel_size, stride, normalize, activation, initialize)

  This function creates a block for a convolutional neural network (CNN) layer.

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param filter: Number of filters.
  :type filter: int
  :param kernel_size: Size of the convolutional kernel.
  :type kernel_size: int
  :param stride: Stride of the convolution.
  :type stride: int
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :return: A tuple containing a sequence of modules representing the CNN block and the updated output shape (C, H, W).
  :rtype: tuple

.. py:function::
  xuance.mindspore.utils.layers.pooling_block(input_shape, scale, pooling)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param scale: Scaling factor for pooling.
  :type scale: float
  :param pooling: Pooling layer (either AdaptiveMaxPool2d or AdaptiveAvgPool2d).
  :type pooling: int
  :return: A sequence of modules representing the pooling block.
  :rtype: list

.. py:function::
  xuance.mindspore.utils.layers.gru_block(input_shape, output_dim, num_layers, dropout, initialize)

  This function creates a block for a Gated Recurrent Unit (GRU) layer.

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of LSTM layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :return: A tuple containing the LSTM module and the output dimension.
  :rtype: tuple

.. py:function::
  xuance.mindspore.utils.layers.lstm_block(input_shape, output_dim, num_layers, dropout, initialize)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param output_dim: the dimension of the output data.
  :type output_dim: int
  :param num_layers: Number of LSTM layers.
  :type num_layers: int
  :param dropout: Dropout probability.
  :type dropout: float
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :return: A tuple containing the LSTM module and the output dimension.
  :rtype: tuple

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import torch
        import torch.nn as nn
        from typing import Optional, Sequence, Tuple, Type, Union, Callable

        ModuleType = Type[nn.Module]


        def mlp_block(input_dim: int,
                      output_dim: int,
                      normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
                      activation: Optional[ModuleType] = None,
                      initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      device: Optional[Union[str, int, torch.device]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
            block = []
            linear = nn.Linear(input_dim, output_dim, device=device)
            if initialize is not None:
                initialize(linear.weight)
                nn.init.constant_(linear.bias, 0)
            block.append(linear)
            if activation is not None:
                block.append(activation())
            if normalize is not None:
                block.append(normalize(output_dim, device=device))
            return block, (output_dim,)


        def cnn_block(input_shape: Sequence[int],
                      filter: int,
                      kernel_size: int,
                      stride: int,
                      normalize: Optional[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d]] = None,
                      activation: Optional[ModuleType] = None,
                      initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      device: Optional[Union[str, int, torch.device]] = None
                      ) -> Tuple[Sequence[ModuleType], Tuple]:
            assert len(input_shape) == 3  # CxHxW
            C, H, W = input_shape
            padding = int((kernel_size - stride) // 2)
            block = []
            cnn = nn.Conv2d(C, filter, kernel_size, stride, padding=padding, device=device)
            if initialize is not None:
                initialize(cnn.weight)
                nn.init.constant_(cnn.bias, 0)
            block.append(cnn)
            C = filter
            H = int((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            W = int((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            if activation is not None:
                block.append(activation())
            if normalize is not None:
                if normalize == nn.GroupNorm:
                    block.append(normalize(C // 2, C, device=device))
                elif normalize == nn.LayerNorm:
                    block.append(normalize((C, H, W), device=device))
                else:
                    block.append(normalize(C, device=device))
            return block, (C, H, W)


        def pooling_block(input_shape: Sequence[int],
                          scale: int,
                          pooling: Union[nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
                          device: Optional[Union[str, int, torch.device]] = None) -> Sequence[ModuleType]:
            assert len(input_shape) == 3  # CxHxW
            block = []
            C, H, W = input_shape
            block.append(pooling(output_size=(H // scale, W // scale), device=device))
            return block


        def gru_block(input_dim: int,
                      output_dim: int,
                      num_layers: int = 1,
                      dropout: float = 0,
                      initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      device: Optional[Union[str, int, torch.device]] = None) -> Tuple[nn.Module, int]:
            gru = nn.GRU(input_size=input_dim,
                         hidden_size=output_dim,
                         num_layers=num_layers,
                         batch_first=True,
                         dropout=dropout,
                         device=device)
            if initialize is not None:
                for weight_list in gru.all_weights:
                    for weight in weight_list:
                        if len(weight.shape) > 1:
                            initialize(weight)
                        else:
                            nn.init.constant_(weight, 0)
            return gru, output_dim


        def lstm_block(input_dim: int,
                       output_dim: int,
                       num_layers: int = 1,
                       dropout: float = 0,
                       initialize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                       device: Optional[Union[str, int, torch.device]] = None) -> Tuple[nn.Module, int]:
            lstm = nn.LSTM(input_size=input_dim,
                           hidden_size=output_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           device=device)
            if initialize is not None:
                for weight_list in lstm.all_weights:
                    for weight in weight_list:
                        if len(weight.shape) > 1:
                            initialize(weight)
                        else:
                            nn.init.constant_(weight, 0)
            return lstm, output_dim

  .. group-tab:: TensorFlow

    .. code-block:: python

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


  .. group-tab:: MindSpore

    .. code-block:: python

        import mindspore as ms
        import mindspore.nn as nn
        from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union, Callable

        ModuleType = Type[nn.Cell]


        def mlp_block(input_dim: int,
                      output_dim: int,
                      normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
                      activation: Optional[ModuleType] = None,
                      initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
                      ) -> Sequence[ModuleType]:
            block = []
            linear = nn.Dense(int(input_dim), int(output_dim))
            if initialize is not None:
                initialize(linear.weight)
            block.append(linear)
            if normalize is not None:
                block.append(normalize(output_dim))
            if activation is not None:
                block.append(activation())
            return block, (output_dim,)


        def cnn_block(input_shape: Sequence[int],
                      filter: int,
                      kernel_size: int,
                      stride: int,
                      normalize: Optional[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d]] = None,
                      activation: Optional[ModuleType] = None,
                      initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
                      ) -> Sequence[ModuleType]:
            assert len(input_shape) == 3  # CxHxW
            C, H, W = input_shape
            padding = int((kernel_size - stride) // 2)
            block = []
            cnn = nn.Conv2d(C, filter, kernel_size, stride, pad_mode="pad", padding=padding)
            if initialize is not None:
                initialize(cnn.weight)
            block.append(cnn)
            C = filter
            H = int((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            W = int((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            if normalize is not None:
                if normalize == nn.GroupNorm:
                    block.append(normalize(C // 2, C))
                elif normalize == nn.LayerNorm:
                    block.append(normalize((C, H, W)))
                else:
                    block.append(normalize(C))
            if activation is not None:
                block.append(activation())
            return block, (C, H, W)


        def pooling_block(input_shape: Sequence[int],
                          scale: int,
                          pooling: Union[nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d]
                          ) -> Sequence[ModuleType]:
            assert len(input_shape) == 3  # CxHxW
            block = []
            C, H, W = input_shape
            block.append(pooling(output_size=(H // scale, W // scale)))
            return block


        def gru_block(input_dim: Sequence[int],
                      output_dim: int,
                      num_layers: int = 1,
                      dropout: float = 0,
                      initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
                      ) -> ModuleType:
            gru = nn.GRU(input_size=input_dim,
                         hidden_size=output_dim,
                         num_layers=num_layers,
                         batch_first=True,
                         dropout=float(dropout)
                         )
            if initialize is not None:
                for weight_list in gru.all_weights:
                    for weight in weight_list:
                        if len(weight.shape) > 1:
                            initialize(weight)
            return gru, output_dim


        def lstm_block(input_dim: Sequence[int],
                       output_dim: int,
                       num_layers: int = 1,
                       dropout: float = 0,
                       initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
                       ) -> ModuleType:
            lstm = nn.LSTM(input_size=input_dim,
                           hidden_size=output_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=float(dropout)
                           )
            if initialize is not None:
                for weight_list in lstm.w_hh_list:
                    for weight in weight_list:
                        if len(weight.shape) > 1:
                            initialize(weight)
                for weight_list in lstm.w_ih_list:
                    for weight in weight_list:
                        if len(weight.shape) > 1:
                            initialize(weight)
            return lstm, output_dim

