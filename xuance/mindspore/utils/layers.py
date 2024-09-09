import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from xuance.common import Optional, Sequence, Tuple, Type, Union, Callable

ModuleType = Type[nn.Cell]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
              ) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    linear = nn.Dense(int(input_dim), int(output_dim))
    if initialize is not None:
        initialize(linear.weight)
        linear.bias.data.set_data(ops.zeros(linear.bias.data.shape))
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        block.append(normalize((output_dim, )))
    return block, (output_dim,)


def cnn_block(input_shape: Sequence[int],
              filter: int,
              kernel_size: int,
              stride: int,
              normalize: Optional[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
              ) -> Tuple[Sequence[ModuleType], Tuple]:
    assert len(input_shape) == 3  # CxHxW
    C, H, W = input_shape
    padding = int((kernel_size - stride) // 2)
    block = []
    cnn = nn.Conv2d(C, filter, kernel_size, stride, pad_mode="pad", padding=padding)
    if initialize is not None:
        initialize(cnn.weight)
        cnn.bias.data.zero_()
        cnn.bias.data.set_data(ops.zeros(cnn.bias.data.shape))
    block.append(cnn)
    C = filter
    H = int((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    W = int((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        if normalize == nn.GroupNorm:
            block.append(normalize(C // 2, C))
        elif normalize == nn.LayerNorm:
            block.append(normalize((C, H, W)))
        else:
            block.append(normalize((C, )))
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


def gru_block(input_dim: int,
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
              ) -> Tuple[nn.Cell, int]:
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
                else:
                    weight.set_value(0)
    return gru, output_dim


def lstm_block(input_dim: int,
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[ms.Tensor], ms.Tensor]] = None
               ) -> Tuple[nn.Cell, int]:
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
                else:
                    weight.set_value(0)
        for weight_list in lstm.w_ih_list:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
                else:
                    weight.set_value(0)
    return lstm, output_dim
