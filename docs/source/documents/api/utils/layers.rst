Neural Network Layers
=======================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**


.. py:function::
  xuance.torch.utils.layers.mlp_block(input_dim, output_dim, normalize, activation, initialize, device)

  xxxxxx.

  :param input_dim: xxxxxx.
  :type input_dim: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.layers.cnn_block(input_shape, filter, kernel_size, stride, normalize, activation, initialize, device)

  xxxxxx.

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param filter: xxxxxx.
  :type filter: xxxxxx
  :param kernel_size: xxxxxx.
  :type kernel_size: xxxxxx
  :param stride: xxxxxx.
  :type stride: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.layers.pooling_block(input_shape, scale, pooling, device)

  xxxxxx.

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param scale: xxxxxx.
  :type scale: xxxxxx
  :param pooling: xxxxxx.
  :type pooling: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.layers.gru_block(input_dim, output_dim, num_layers, dropout, initialize)

  xxxxxx.

  :param input_dim: xxxxxx.
  :type input_dim: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param num_layers: xxxxxx.
  :type num_layers: xxxxxx
  :param dropout: xxxxxx.
  :type dropout: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.layers.lstm_block(input_dim, output_dim, num_layers, dropout, initialize, device)

  xxxxxx.

  :param input_dim: xxxxxx.
  :type input_dim: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param num_layers: xxxxxx.
  :type num_layers: xxxxxx
  :param dropout: xxxxxx.
  :type dropout: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:function::
  xuance.mindspore.utils.layers.mlp_block(input_dim, output_dim, normalize, activation, initialize)

  :param input_dim: xxxxxx.
  :type input_dim: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.layers.cnn_block(input_shape, filter, kernel_size, stride, normalize, activation, initialize)

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param filter: xxxxxx.
  :type filter: xxxxxx
  :param kernel_size: xxxxxx.
  :type kernel_size: xxxxxx
  :param stride: xxxxxx.
  :type stride: xxxxxx
  :param normalize: xxxxxx.
  :type normalize: xxxxxx
  :param activation: xxxxxx.
  :type activation: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.layers.pooling_block(input_shape, scale, pooling)

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param scale: xxxxxx.
  :type scale: xxxxxx
  :param pooling: xxxxxx.
  :type pooling: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.layers.gru_block(input_shape, output_dim, num_layers, dropout, initialize)

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param num_layers: xxxxxx.
  :type num_layers: xxxxxx
  :param dropout: xxxxxx.
  :type dropout: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.layers.lstm_block(input_shape, output_dim, num_layers, dropout, initialize)

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param output_dim: xxxxxx.
  :type output_dim: xxxxxx
  :param num_layers: xxxxxx.
  :type num_layers: xxxxxx
  :param dropout: xxxxxx.
  :type dropout: xxxxxx
  :param initialize: xxxxxx.
  :type initialize: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

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

