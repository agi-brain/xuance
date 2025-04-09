"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn

ModuleType = Optional[Type[nn.Module]]
ArgType = Union[Tuple[Any, ...], Dict[Any, Any], None]
ArgsType = Union[ArgType, List[ArgType]]


def create_layer_with_args(layer_type: ModuleType, layer_args: Optional[ArgType]) -> nn.Module:
    """Create a single layer with given layer type and arguments.

    Args:
        layer_type (ModuleType): the type of the layer to be created.
        layer_args (ArgType, optional): the arguments to be passed to the layer.
    """
    if layer_type is None:
        raise ValueError("`layer_type` must be not None")
    if isinstance(layer_args, tuple):
        return layer_type(*layer_args)
    elif isinstance(layer_args, dict):
        return layer_type(**layer_args)
    elif layer_args is None:
        return layer_type()
    else:
        raise ValueError(f"`layer_args` must be None, tuple or dict, got {type(layer_args)}")


def miniblock(
    input_size: int,
    output_size: int,
    layer_type: Type[nn.Module] = nn.Linear,
    layer_args: ArgType = None,
    dropout_layer: ModuleType = None,
    dropout_args: ArgType = None,
    norm_layer: ModuleType = None,
    norm_args: ArgType = None,
    activation: ModuleType = None,
    act_args: ArgType = None,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, dropout layer, norm layer and activation function.

    Based on Tianshou's miniblock function
    (https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py).

    Args:
        input_size (int): the input size of the miniblock (in_features for Linear and in_channels for Conv2d).
        output_size (int): the output size of the miniblock.
        layer_type (Type[nn.Linear], optional): the type of the layer to be created. Defaults to nn.Linear.
        layer_args (ArgType, optional): the arguments to be passed to the layer.
            Defaults to None.
        dropout_layer (ModuleType, optional): the type of the dropout layer to be created. Defaults to None.
        dropout_args (ArgType, optional): the arguments to be passed to the dropout
            layer. Defaults to None.
        norm_layer (ModuleType, optional): the type of the norm layer to be created. Defaults to None.
        norm_args (ArgType, optional): the arguments to be passed to the norm layer.
            Defaults to None.
        activation (ModuleType, optional): the type of the activation function to be created.
            Defaults to None.
        act_args (Tuple[Any, ...] | Dict[Any, Any] | None, optional): the arguments to be passed to the activation
            function. Defaults to None.

    Returns:
        List[nn.Module]: the miniblock as a list of layers.
    """
    if layer_args is None:
        layers: List[nn.Module] = [layer_type(input_size, output_size)]
    elif isinstance(layer_args, tuple):
        layers = [layer_type(input_size, output_size, *layer_args)]
    elif isinstance(layer_args, dict):
        layers = [layer_type(input_size, output_size, **layer_args)]
    else:
        raise ValueError(f"layer_args must be None, tuple or dict, got {type(layer_args)}")

    if dropout_layer is not None:
        layers += [create_layer_with_args(dropout_layer, dropout_args)]

    if norm_layer is not None:
        layers += [create_layer_with_args(norm_layer, norm_args)]

    if activation is not None:
        layers += [create_layer_with_args(activation, act_args)]
    return layers


def create_layers(
    layer_type: Union[ModuleType, List[ModuleType]], layer_args: Optional[ArgsType], num_layers: int
) -> Tuple[List[ModuleType], ArgsType]:
    """Create a list of layers with given layer type and arguments.

    If a layer_type is not specified, then the lists will be filled with None. If the layer type or the layer arguments
    are specified only once, they will be cast to a sequence of length num_layers.

    Args:
        layer_type (Union[ModuleType, Sequence[ModuleType]]): the type of the layer to be created.
        layer_args (ArgsType, optional): the arguments to be passed to the layer.
        num_layers (int): the number of layers to be created.

    Returns:
        Tuple[Sequence[ModuleType], ArgsType]: a list of layers and a list of args.

    Examples:
        >>> create_layers(nn.Linear, None, 3)
        ([nn.Linear, nn.Linear, nn.Linear], [None, None, None])

        >>> create_layers(nn.Linear, {"arg1":3, "arg2": "foo"}, 3)
        (
            [nn.Linear, nn.Linear, nn.Linear],
            [{'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}, {'arg1': 3, 'arg2': 'foo'}]
        )

        >>> create_layers([nn.Linear, nn.Conv2d], [{"bias":False}, {"kernel_size": 5, "bias": True}], 2)
        ([nn.Linear, nn.Conv2d], [{'bias': False}, {'kernel_size':5, 'bias': True}])

        >>> create_layers([nn.Linear, nn.Linear], (64, 10), 2)
        ([nn.Linear, nn.Linear], [(64, 10), (64, 10)])
    """
    if layer_type is None:
        layers_list = [None] * num_layers
        args_list = [None] * num_layers
        return layers_list, args_list

    if isinstance(layer_type, list):
        assert len(layer_type) == num_layers
        layers_list = layer_type
        if isinstance(layer_args, list):
            assert len(layer_args) == num_layers
            args_list = layer_args
        else:
            args_list = [layer_args for _ in range(num_layers)]
    else:
        layers_list = [layer_type for _ in range(num_layers)]
        args_list = [layer_args for _ in range(num_layers)]
    return layers_list, args_list


def per_layer_ortho_init_weights(module: nn.Module, gain: float = 1.0, bias: float = 0.0):
    """Initialize the weights of a module with orthogonal weights.

    Args:
        module (nn.Module): module to initialize
        gain (float, optional): gain of the orthogonal initialization. Defaults to 1.0.
        bias (float, optional): bias of the orthogonal initialization. Defaults to 0.0.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, val=bias)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=gain)
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for i in range(len(module)):
            per_layer_ortho_init_weights(module[i], gain=gain, bias=bias)


def cnn_forward(
    model: nn.Module,
    input: Tensor,
    input_dim: Union[torch.Size, Tuple[int, ...]],
    output_dim: Union[torch.Size, Tuple[int, ...]],
) -> Tensor:
    """
    Compute the forward of a Convolutional neural network.
    It flattens all the dimensions before the model input_size, i.e.,
    the dimensions before the (C_in, H, W) dimensions for the encoder
    and the dimensions before the (feature_size,) dimension for the decoder.

    Args:
        model (nn.Module): the model.
        input (Tensor): the input tensor of dimension (*, C_in, H, W) or (*, feature_size),
            where * means any number of dimensions including None.
        input_dim (Union[torch.Size, Tuple[int, ...]]): the input dimensions,
            i.e., either (C_in, H, W) or (feature_size,).
        output_dim (Union[torch.Size, Tuple[int, ...]]): the desired dimensions in output.

    Returns:
        The output of dimensions (*, *output_dim).

    Examples:
        >>> encoder
        CNN(
            (network): Sequential(
                (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU()
                (4): Flatten(start_dim=1, end_dim=-1)
                (5): Linear(in_features=128, out_features=25, bias=True)
            )
        )
        >>> input = torch.rand(10, 20, 3, 4, 4)
        >>> cnn_forward(encoder, input, (3, 4, 4), -1).shape
        torch.Size([10, 20, 25])

        >>> decoder
        Sequential(
            (0): Linear(in_features=230, out_features=1024, bias=True)
            (1): Unflatten(dim=-1, unflattened_size=(1024, 1, 1))
            (2): ConvTranspose2d(1024, 128, kernel_size=(5, 5), stride=(2, 2))
            (3): ReLU()
            (4): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2))
            (5): ReLU()
            (6): ConvTranspose2d(64, 32, kernel_size=(6, 6), stride=(2, 2))
            (7): ReLU()
            (8): ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(2, 2))
        )
        >>> input = torch.rand(10, 20, 230)
        >>> cnn_forward(decoder, input, (230,), (3, 64, 64)).shape
        torch.Size([10, 20, 3, 64, 64])
    """
    batch_shapes = input.shape[: -len(input_dim)]
    flatten_input = input.reshape(-1, *input_dim)
    model_out = model(flatten_input)
    return model_out.reshape(*batch_shapes, *output_dim)


"""
Adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py
"""

import warnings
from math import prod
from typing import Any, Callable, Dict, Optional, Sequence, Union, no_type_check

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    """Simple MLP backbone.

    Args:
        input_dims (Union[int, Sequence[int]]): dimensions of the input vector.
        output_dim (int, optional): dimension of the output vector. If set to None, there
            is no final linear layer. Else, a final linear layer is added.
            Defaults to None.
        hidden_sizes (Sequence[int], optional): shape of MLP passed in as a list, not including
            input_dims and output_dim.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
        flatten_dim (int, optional): whether to flatten input data. The flatten dimension starts from 1.
            Defaults to True.
    """

    def __init__(
        self,
        input_dims: Union[int, Sequence[int]],
        output_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (),
        layer_args: Optional[ArgsType] = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        flatten_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_sizes)
        if num_layers < 1 and output_dim is None:
            raise ValueError("The number of layers should be at least 1.")

        if isinstance(input_dims, Sequence) and flatten_dim is None:
            warnings.warn(
                "input_dims is a sequence, but flatten_dim is not specified. "
                "Be careful to flatten the input data correctly before the forward."
            )

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        if isinstance(input_dims, int):
            input_dims = [input_dims]
        hidden_sizes = [prod(input_dims)] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, nn.Linear, l_args, drop, drop_args, norm, norm_args, activ, act_args)
        if output_dim is not None:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]

        self._output_dim = output_dim or hidden_sizes[-1]
        self._model = nn.Sequential(*model)
        self._flatten_dim = flatten_dim

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def flatten_dim(self) -> Optional[int]:
        return self._flatten_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        if self.flatten_dim is not None:
            obs = obs.flatten(self.flatten_dim)
        return self.model(obs)


class CNN(nn.Module):
    """Simple CNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        cnn_layer: ModuleType = nn.Conv2d,
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)


class DeCNN(nn.Module):
    """Simple DeCNN backbone.

    Args:
        input_channels (int): dimensions of the input channels.
        hidden_channels (Sequence[int], optional): intermediate number of channels for the CNN,
            including the output channels.
        dropout_layer (Union[ModuleType, Sequence[ModuleType]], optional): which dropout layer to be used
            before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
            You can also pass a list of dropout modules with the same length
            of hidden_sizes to use different dropout modules in different layers.
            If None, then no dropout layer is used.
            Defaults to None.
        norm_layer (Union[ModuleType, Sequence[ModuleType]], optional): which normalization layer to be used
            before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes to use different normalization modules in different layers.
            If None, then no normalization layer is used.
            Defaults to None.
        activation (Union[ModuleType, Sequence[ModuleType]], optional): which activation to use after each layer,
            can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
            activations for different layers if a list is passed.
            Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int] = (),
        cnn_layer: ModuleType = nn.ConvTranspose2d,
        layer_args: ArgsType = None,
        dropout_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        dropout_args: Optional[ArgsType] = None,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
    ) -> None:
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)


class NatureCNN(CNN):
    """CNN from DQN Nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.

    Args:
        in_channels (int): the input channels to the first convolutional layer
        features_dim (int): the features dimension in output from the last convolutional layer
        screen_size (int, optional): the dimension of the input image as a single integer.
            Needed to extract the features and compute the output dimension after all the
            convolutional layers.
            Defaults to 64.
    """

    def __init__(self, in_channels: int, features_dim: int, screen_size: int = 64):
        super().__init__(
            in_channels,
            [32, 64, 64],
            layer_args=[
                {"kernel_size": 8, "stride": 4},
                {"kernel_size": 4, "stride": 2},
                {"kernel_size": 3, "stride": 1},
            ],
        )

        with torch.no_grad():
            x = self.model(torch.rand(1, in_channels, screen_size, screen_size, device=self.model[0].weight.device))
            out_dim = x.flatten(1).shape[1]
        self._output_dim = out_dim
        self.fc = None
        if features_dim is not None:
            self._output_dim = features_dim
            self.fc = nn.Linear(out_dim, features_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = cnn_forward(self.model, x, input_dim=x.shape[-3:], output_dim=(-1,))
        x = F.relu(self.fc(x))
        return x


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to nn.Identiy.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {}.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.Identity,
        layer_norm_kw: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


class MultiEncoder(nn.Module):
    def __init__(
        self,
        cnn_encoder: ModuleType,
        mlp_encoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_encoder is None and mlp_encoder is None:
            raise ValueError("There must be at least one encoder, both cnn and mlp encoders are None")
        self.has_cnn_encoder = False
        self.has_mlp_encoder = False
        if cnn_encoder is not None:
            if getattr(cnn_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(cnn_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`cnn_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_cnn_encoder = True
        if mlp_encoder is not None:
            if getattr(mlp_encoder, "input_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `input_dim` attribute representing "
                    "the dimension of the input tensor"
                )
            if getattr(mlp_encoder, "output_dim", None) is None:
                raise AttributeError(
                    "`mlp_encoder` must contain the `output_dim` attribute representing "
                    "the dimension of the output tensor"
                )
            self.has_mlp_encoder = True
        self.has_both_encoders = self.has_cnn_encoder and self.has_mlp_encoder
        self.cnn_encoder = cnn_encoder
        self.mlp_encoder = mlp_encoder
        self.cnn_input_dim = self.cnn_encoder.input_dim if self.cnn_encoder is not None else None
        self.mlp_input_dim = self.mlp_encoder.input_dim if self.mlp_encoder is not None else None
        self.cnn_output_dim = self.cnn_encoder.output_dim if self.cnn_encoder is not None else 0
        self.mlp_output_dim = self.mlp_encoder.output_dim if self.mlp_encoder is not None else 0
        self.output_dim = self.cnn_output_dim + self.mlp_output_dim

    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_encoder.keys if self.cnn_encoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_encoder.keys if self.mlp_encoder is not None else []

    def forward(self, obs: Dict[str, Tensor], *args, **kwargs) -> Tensor:
        if self.has_cnn_encoder:
            cnn_out = self.cnn_encoder(obs, *args, **kwargs)
        if self.has_mlp_encoder:
            mlp_out = self.mlp_encoder(obs, *args, **kwargs)
        if self.has_both_encoders:
            return torch.cat((cnn_out, mlp_out), -1)
        elif self.has_cnn_encoder:
            return cnn_out
        else:
            return mlp_out


class MultiDecoder(nn.Module):
    def __init__(
        self,
        cnn_decoder: ModuleType,
        mlp_decoder: ModuleType,
    ) -> None:
        super().__init__()
        if cnn_decoder is None and mlp_decoder is None:
            raise ValueError("There must be an decoder, both cnn and mlp decoders are None")
        self.cnn_decoder = cnn_decoder
        self.mlp_decoder = mlp_decoder

    @property
    def cnn_keys(self) -> Sequence[str]:
        return self.cnn_decoder.keys if self.cnn_decoder is not None else []

    @property
    def mlp_keys(self) -> Sequence[str]:
        return self.mlp_decoder.keys if self.mlp_decoder is not None else []

    # adapt to xuance: the return Dict[str, Tensor] modified to -> Tensor
    def forward(self, x: Tensor) -> Tensor:
        reconstructed_obs = None
        if self.cnn_decoder is not None:
            reconstructed_obs = self.cnn_decoder(x)[0]
        if self.mlp_decoder is not None:
            reconstructed_obs = self.mlp_decoder(x)
        return reconstructed_obs


class LayerNormChannelLast(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {len(x.shape)}D instead: {x.shape}")
        input_dtype = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.to(input_dtype)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        out = super().forward(x)
        return out.to(input_dtype)



