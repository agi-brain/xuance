import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from xuance.torch.utils.layers4dreamder import (
    CNN,
    MLP,
    DeCNN,
    LayerNorm,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
    ModuleType,
    cnn_forward
)
from xuance.torch.utils import sym_log, dotdict, init_weights, uniform_init_weights, compute_stochastic_state


class CNNEncoder(nn.Module):
    """The Dreamer-V3 image encoder. This is composed of 4 `nn.Conv2d` with
    kernel_size=3, stride=2 and padding=1. No bias is used if a `nn.LayerNorm`
    is used after the convolution. This 4-stages model assumes that the image
    is a 64x64 and it ends with a resolution of 4x4. If more than one image is to be encoded, then those will
    be concatenated on the channel dimension and fed to the encoder.

    Args:
        input_channels (Sequence[int]): the input channels, one for each image observation to encode.
        image_size (Tuple[int, int]): the image size as (Height,Width).
        channels_multiplier (int): the multiplier for the output channels. Given the 4 stages, the 4 output channels
            will be [1, 2, 4, 8] * `channels_multiplier`.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNormChannelLast.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function.
            Defaults to nn.SiLU.
        stages (int, optional): how many stages for the CNN.
    """

    def __init__(
            self,
            input_channels: Sequence[int],
            image_size: Tuple[int, int],
            channels_multiplier: int,
            layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
            activation: ModuleType = nn.SiLU,
            stages: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = (sum(input_channels), *image_size)
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_dim[0],
                hidden_channels=(torch.tensor([2 ** i for i in range(stages)]) * channels_multiplier).tolist(),
                cnn_layer=nn.Conv2d,
                layer_args={"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity},
                activation=activation,
                norm_layer=[layer_norm_cls] * stages,
                norm_args=[
                    {**layer_norm_kw, "normalized_shape": (2 ** i) * channels_multiplier} for i in range(stages)
                ],
            ),
            nn.Flatten(-3, -1),
        )
        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]

    def forward(self, obs: Tensor) -> Tensor:
        return cnn_forward(self.model, obs, obs.shape[-3:], (-1,))


class CNNDecoder(nn.Module):
    """The exact inverse of the `CNNEncoder` class. It assumes an initial resolution
    of 4x4, and in 4 stages reconstructs the observation image to 64x64. If multiple
    images are to be reconstructed, then it will create a dictionary with an entry
    for every reconstructed image. No bias is used if a `nn.LayerNorm` is used after
    the `nn.Conv2dTranspose` layer.

    Args:
        output_channels (Sequence[int]): the output channels, one for every image observation.
        channels_multiplier (int): the channels multiplier, same for the encoder network.
        latent_state_size (int): the size of the latent state. Before applying the decoder,
            a `nn.Linear` layer is used to project the latent state to a feature vector
            of dimension [8 * `channels_multiplier`, 4, 4].
        cnn_encoder_output_dim (int): the output of the image encoder. It should be equal to
            8 * `channels_multiplier` * 4 * 4.
        image_size (Tuple[int, int]): the final image size.
        activation (nn.Module, optional): the activation function.
            Defaults to nn.SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNormChannelLast.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        stages (int): how many stages in the CNN decoder.
    """

    def __init__(
            self,
            output_channels: Sequence[int],
            channels_multiplier: int,
            latent_state_size: int,
            cnn_encoder_output_dim: int,
            image_size: Tuple[int, int],
            activation: nn.Module = nn.SiLU,
            layer_norm_cls: Callable[..., nn.Module] = LayerNormChannelLast,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
            stages: int = 4,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.cnn_encoder_output_dim = cnn_encoder_output_dim
        self.image_size = image_size
        self.output_dim = (sum(output_channels), *image_size)
        self.model = nn.Sequential(
            nn.Linear(latent_state_size, cnn_encoder_output_dim),
            nn.Unflatten(1, (-1, 4, 4)),
            DeCNN(
                input_channels=(2 ** (stages - 1)) * channels_multiplier,
                hidden_channels=(
                                        torch.tensor(
                                            [2 ** i for i in reversed(range(stages - 1))]) * channels_multiplier
                                ).tolist()
                                + [self.output_dim[0]],
                cnn_layer=nn.ConvTranspose2d,
                layer_args=[
                               {"kernel_size": 4, "stride": 2, "padding": 1, "bias": layer_norm_cls == nn.Identity}
                               for _ in range(stages - 1)
                           ]
                           + [{"kernel_size": 4, "stride": 2, "padding": 1}],
                activation=[activation for _ in range(stages - 1)] + [None],
                norm_layer=[layer_norm_cls for _ in range(stages - 1)] + [None],
                norm_args=[
                              {**layer_norm_kw, "normalized_shape": (2 ** (stages - i - 2)) * channels_multiplier}
                              for i in range(stages - 1)
                          ]
                          + [None],
            ),
        )

    def forward(self, latent_states: Tensor) -> List[Tensor]:
        cnn_out = cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_dim)
        return torch.split(cnn_out, self.output_channels, -3)


class MLPEncoder(nn.Module):
    """The Dreamer-V3 vector encoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be encoded, then those will concatenated on the last
    dimension before being fed to the encoder.

    Args:
        input_dims (Sequence[int]): the dimensions of every vector to encode.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.SiLU.
        sym_log_inputs (bool, optional): whether to squash the input with the sym_log function.
            Defaults to True.
    """

    def __init__(
            self,
            input_dims: Sequence[int],
            mlp_layers: int = 4,
            dense_units: int = 512,
            layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
            activation: ModuleType = nn.SiLU,
            sym_log_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = sum(input_dims)  # [4]
        self.model = MLP(
            self.input_dim,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.output_dim = dense_units
        self.sym_log_inputs = sym_log_inputs  # True

    def forward(self, obs: Tensor) -> Tensor:
        # x = torch.cat([sym_log(obs[k]) if self.sym_log_inputs else obs[k] for k in self.keys], -1)
        return self.model(sym_log(obs))


class MLPDecoder(nn.Module):
    """The exact inverse of the MLPEncoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be decoded, then it will create a dictionary with an entry
    for every reconstructed vector.

    Args:
        keys (Sequence[str]): the keys representing the vector observations to decode.
        output_dims (Sequence[int]): the dimensions of every vector to decode.
        latent_state_size (int): the dimension of the latent state.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.SiLU.
    """

    def __init__(
        self,
        output_dims: Sequence[int],
        latent_state_size: int,
        mlp_layers: int = 4,
        dense_units: int = 512,
        activation: ModuleType = nn.SiLU,
        layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
        layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.output_dims = output_dims
        self.model = MLP(
            latent_state_size,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        x = self.model(latent_states)
        return self.heads[0](x)  # revised to adapt to xuance


class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.

    Args:
        input_size (int): the input size of the model.
        dense_units (int): the number of dense units.
        recurrent_state_size (int): the size of the recurrent state.
        activation_fn (nn.Module): the activation function.
            Default to SiLU.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
    """

    def __init__(
            self,
            input_size: int,
            recurrent_state_size: int,
            dense_units: int,
            activation_fn: nn.Module = nn.SiLU,
            layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=[layer_norm_cls],
            norm_args=[{**layer_norm_kw, "normalized_shape": dense_units}],
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=False,
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
        )
        self.recurrent_state_size = recurrent_state_size

    def forward(self, input: Tensor, recurrent_state: Tensor) -> Tensor:
        """
        Compute the next recurrent state from the latent state (stochastic and recurrent states) and the actions.

        Args:
            input (Tensor): the input tensor composed by the stochastic state and the actions concatenated together.
            recurrent_state (Tensor): the previous recurrent state.

        Returns:
            the computed recurrent output and recurrent state.
        """
        feat = self.mlp(input)
        out = self.rnn(feat, recurrent_state)
        return out


class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
            [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a
            multi-layer perceptron to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multi-layer perceptron to predict the stochastic part of the latent state.
        distribution_config (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over states, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`, where `unif = `1 / self.discrete`.
            Defaults to 0.01.
    """

    def __init__(
            self,
            recurrent_model: RecurrentModel,
            representation_model: nn.Module,
            transition_model: nn.Module,
            distribution_config: Dict[str, Any],
            discrete: int = 32,
            unimix: float = 0.01,
            learnable_initial_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_config = distribution_config
        self.discrete = discrete
        self.unimix = unimix
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state", torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )

    def get_initial_states(self, batch_shape: Union[Sequence[int], torch.Size]) -> Tuple[Tensor, Tensor]:
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    def dynamic(
            self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the prior from the recurrent output.
            Representation model: compute the posterior from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
        and [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

        Args:
            posterior (Tensor): the stochastic state computed by the representation model (posterior). It is expected
                to be of dimension `[stoch_size, self.discrete]`, which by default is `[32, 32]`.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.
            is_first (Tensor): if this is the first step in the episode.

        Returns:
            The recurrent state (Tensor): the recurrent state of the recurrent model.
            The posterior stochastic state (Tensor): computed by the representation model
            The prior stochastic state (Tensor): computed by the transition model
            The logits of the posterior state (Tensor): computed by the transition model from the recurrent state.
            The logits of the prior state (Tensor): computed by the transition model from the recurrent state.
            from the recurrent state and the embbedded observation.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(self, recurrent_state: Tensor, embedded_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: Tensor = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor, sample_state=True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits: Tensor = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)

    def imagination(self, prior: Tensor, recurrent_state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        distribution_config (Dict[str, Any]): The configs of the distributions.
        init_std (float): the amount to sum to the standard deviation.
            Default to 0.0.
        min_std (float): the minimum standard deviation for the actions.
            Default to 1.0.
        max_std (float): the maximum standard deviation for the actions.
            Default to 1.0.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 1024.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.SiLU.
        mlp_layers (int): the number of dense layers.
            Default to 5.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to LayerNorm.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {"eps": 1e-3}.
        unimix: (float, optional): the percentage of uniform distribution to inject into the categorical
            distribution over actions, i.e. given some logits `l` and probabilities `p = softmax(l)`,
            then `p = (1 - self.unimix) * p + self.unimix * unif`,
            where `unif = `1 / self.discrete`.
            Defaults to 0.01.
        action_clip (float): the action clip parameter.
            Default to 1.0.
    """

    def __init__(
            self,
            latent_state_size: int,
            actions_dim: Sequence[int],
            is_continuous: bool,
            distribution_config: Dict[str, Any],
            init_std: float = 0.0,
            min_std: float = 1.0,
            max_std: float = 1.0,
            dense_units: int = 1024,
            activation: nn.Module = nn.SiLU,
            mlp_layers: int = 5,
            layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
            layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
            unimix: float = 0.01,
            action_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.distribution_config = distribution_config
        self.distribution = distribution_config.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "scaled_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `scaled_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "scaled_normal"
            else:
                self.distribution = "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            layer_args={"bias": layer_norm_cls == nn.Identity},
            norm_layer=layer_norm_cls,
            norm_args={**layer_norm_kw, "normalized_shape": dense_units},
        )
        if is_continuous:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, np.sum(actions_dim) * 2)])
        else:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self._unimix = unimix
        self._action_clip = action_clip

    def forward(
            self, state: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Sequence[Distribution]]:
        """
        Call the forward method of the actor model and reorganizes the result with shape (batch_size, *, num_actions),
        where * means any number of dimensions including None.

        Args:
            state (Tensor): the current state of shape (batch_size, *, stochastic_size + recurrent_state_size).
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Dict[str, Tensor], optional): the mask to use on the actions.
                Default to None.

        Returns:
            The tensor of the actions taken by the agent with shape (batch_size, *, num_actions).
            The distribution of the actions
        """
        out: Tensor = self.model(state)
        pre_dist: List[Tensor] = [head(out) for head in self.mlp_heads]
        if self.is_continuous:
            mean, std = torch.chunk(pre_dist[0], 2, -1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = F.softplus(std + self.init_std) + self.min_std
                actions_dist = Normal(mean, std)
                actions_dist = Independent(TransformedDistribution(actions_dist, TanhTransform()), 1)
            elif self.distribution == "normal":
                actions_dist = Normal(mean, std)
                actions_dist = Independent(actions_dist, 1)
            elif self.distribution == "scaled_normal":
                std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
                dist = Normal(torch.tanh(mean), std)
                actions_dist = Independent(dist, 1)
            else:
                actions_dist = None
            if not greedy:
                actions = actions_dist.rsample()
            else:
                actions = actions_dist.mode
            if self._action_clip > 0.0:
                action_clip = torch.full_like(actions, self._action_clip)
                actions = actions * (action_clip / torch.maximum(action_clip, torch.abs(actions))).detach()
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_dist = []
            actions = []
            for logits in pre_dist:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=self._uniform_mix(logits)))
                if not greedy:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        if self._unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logits = probs_to_logits(probs)
        return logits


class PlayerDV3(nn.Module):
    """
    The model of the Dreamer_v3 player.

    Args:
        encoder (MultiEncoder): the encoder.
        rssm (RSSM): the RSSM model.
        actor (Module): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        transition_model (Module): the transition model.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the player is using ('task' or 'exploration').
            Default to None.
         (bool, optional): whether to use the DecoupledRSSM model.
    """

    def __init__(
            self,
            encoder: MultiEncoder,
            rssm: RSSM,
            actor: Actor,
            actions_dim: Sequence[int],
            num_envs: int,
            stochastic_size: int,
            recurrent_state_size: int,
            device: torch.device,
            discrete_size: int = 32,
            actor_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type

        self.actions, self.recurrent_state, self.stochastic_state = [None] * 3

    @torch.no_grad()
    def init_states(self,
                    reset_envs: Optional[Sequence[int]] = None,
                    num_envs: Optional[int] = None) -> None:
        """Initialize the states and the actions for the ended environments.

        Args:
            reset_envs (Optional[Sequence[int]], optional): which environments' states to reset.
                If None, then all environments' states are reset.
                Defaults to None.
            num_envs (Optional[int]): the number of environments.
                If None, then it will be self.num_envs  # prop added to deal with xuance test
        """
        num_envs = num_envs if num_envs else self.num_envs  # added to deal with xuance test
        if reset_envs is None or len(reset_envs) == 0:  # reset all
            self.actions = torch.zeros(1, num_envs, np.sum(self.actions_dim), device=self.device)
            self.recurrent_state, stochastic_state = self.rssm.get_initial_states((1, num_envs))
            self.stochastic_state = stochastic_state.reshape(1, num_envs, -1)
        else:
            self.actions[:, reset_envs] = torch.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs], stochastic_state = self.rssm.get_initial_states((1, len(reset_envs)))
            self.stochastic_state[:, reset_envs] = stochastic_state.reshape(1, len(reset_envs), -1)

    def get_actions(
            self,
            obs: Dict[str, Tensor],
            greedy: bool = False,
            mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        """
        Return the greedy actions.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            greedy (bool): whether or not to sample the actions.
                Default to False.
            mask (Optional[Dict[str, Tensor]]): action mask
        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.rssm.recurrent_model(  # -> h1: [1: batch, 4: envs, 512]
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


class WorldModel(nn.Module):
    """
    Wrapper class for the World model.

    Args:
        encoder (Module): the encoder.
        rssm (RSSM): the rssm.
        observation_model (Module): the observation model.
        reward_model (Module): the reward model.
        continue_model (Module, optional): the continue model.
    """

    def __init__(
            self,
            encoder,
            rssm: RSSM,
            observation_model,
            reward_model,
            continue_model,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model


class DreamerV3WorldModel(nn.Module):
    def __init__(self, actions_dim: Sequence[int],
                 is_continuous: bool,
                 config: Dict[str, Any],
                 obs_space: gym.spaces.Dict):
        super().__init__()
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.config = config
        self.obs_space = obs_space
        """
        for policy: world_model, actor, critic, target_critic
        for agent: player (link to policy.world_model.~ & policy.actor)
        """
        if self.config.pixel:
            self.obs_space = gym.spaces.Box(0, 255, ((self.obs_space.shape[2], ) + self.obs_space.shape[:2]), np.uint8)
        self.world_model, self.actor, self.critic, self.target_critic, self.player = (
            DreamerV3WorldModel._build_model(
                self.actions_dim,
                self.is_continuous,
                self.config,
                self.obs_space
            )
        )

    @staticmethod
    def _build_model(
        actions_dim: Sequence[int],
        is_continuous: bool,
        config: Dict[str, Any],
        obs_space: gym.spaces.Dict,
    ) -> Tuple[WorldModel, Actor, nn.Module, nn.Module, PlayerDV3]:
        # world_model, actor, critic, target_critic, player
        """Build the models

        Args:
            actions_dim (Sequence[int]): the dimension of the actions.
            is_continuous (bool): whether or not the actions are continuous.
            config (DictConfig): the configs of DreamerV3.
            obs_space (Dict[str, Any]): the observation space.

        Returns:
            The world model (WorldModel): composed by the encoder, rssm, observation and
            reward models and the continue model.
            The actor (nn.Module).
            The critic (nn.Module).
            The target critic (nn.Module).
        """

        config = dotdict(vars(config))
        world_model_config = config.world_model
        actor_config = config.actor
        critic_config = config.critic
        """deter_size; stoch_size; model_state_size"""
        # Sizes
        recurrent_state_size = world_model_config.recurrent_model.recurrent_state_size
        stochastic_size = world_model_config.stochastic_size * world_model_config.discrete_size
        latent_state_size = stochastic_size + recurrent_state_size
        # Define models
        cnn_stages = int(np.log2(config.env_config.screen_size) - np.log2(4))  # 4
        cnn_encoder = (
            CNNEncoder(
                input_channels=[int(np.prod(obs_space.shape[:-2]))],
                image_size=obs_space.shape[-2:],
                channels_multiplier=world_model_config.encoder.cnn_channels_multiplier,
                layer_norm_cls=LayerNormChannelLast,
                layer_norm_kw=world_model_config.encoder.cnn_layer_norm.kw,
                activation=nn.SiLU,
                stages=cnn_stages,
            )
            if config.pixel else None
        )
        mlp_encoder = (
            MLPEncoder(
                input_dims=[obs_space.shape[0]],
                mlp_layers=world_model_config.encoder.mlp_layers,
                dense_units=world_model_config.encoder.dense_units,
                activation=nn.SiLU,
                layer_norm_cls=LayerNorm,
                layer_norm_kw=world_model_config.encoder.mlp_layer_norm.kw,
            )
            if not config.pixel else None
        )
        encoder = MultiEncoder(cnn_encoder, mlp_encoder).to(config.device)

        recurrent_model = RecurrentModel(
            input_size=int(sum(actions_dim) + stochastic_size),
            recurrent_state_size=world_model_config.recurrent_model.recurrent_state_size,
            dense_units=world_model_config.recurrent_model.dense_units,
            layer_norm_cls=LayerNorm,
            layer_norm_kw=world_model_config.recurrent_model.layer_norm.kw,
        )
        represention_model_input_size = encoder.output_dim
        represention_model_input_size += recurrent_state_size
        representation_ln_cls = LayerNorm
        representation_model = MLP(
            input_dims=represention_model_input_size,  # (h1, x1) -> z1
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.representation_model.hidden_size],
            activation=nn.SiLU,
            layer_args={"bias": representation_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=[representation_ln_cls],
            norm_args=[
                {
                    **world_model_config.representation_model.layer_norm.kw,
                    "normalized_shape": world_model_config.representation_model.hidden_size,
                }
            ],
        )
        transition_ln_cls = LayerNorm
        transition_model = MLP(
            input_dims=recurrent_state_size,
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.transition_model.hidden_size],
            activation=nn.SiLU,
            layer_args={"bias": transition_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=[transition_ln_cls],
            norm_args=[
                {
                    **world_model_config.transition_model.layer_norm.kw,
                    "normalized_shape": world_model_config.transition_model.hidden_size,
                }
            ],
        )

        rssm_cls = RSSM
        rssm = rssm_cls(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            transition_model=transition_model.apply(init_weights),
            distribution_config=config.distribution,
            discrete=world_model_config.discrete_size,
            unimix=config.unimix,
            learnable_initial_recurrent_state=config.world_model.learnable_initial_recurrent_state,
        ).to(config.device)

        cnn_decoder = (
            CNNDecoder(
                output_channels=[int(np.prod(obs_space.shape[:-2]))],
                channels_multiplier=world_model_config.observation_model.cnn_channels_multiplier,
                latent_state_size=latent_state_size,
                cnn_encoder_output_dim=cnn_encoder.output_dim,
                image_size=obs_space.shape[-2:],
                activation=nn.SiLU,
                layer_norm_cls=LayerNormChannelLast,
                layer_norm_kw=world_model_config.observation_model.mlp_layer_norm.kw,
                stages=cnn_stages,
            )
            if config.pixel else None
        )
        mlp_decoder = (
            MLPDecoder(
                output_dims=[obs_space.shape[0]],
                latent_state_size=latent_state_size,
                mlp_layers=world_model_config.observation_model.mlp_layers,
                dense_units=world_model_config.observation_model.dense_units,
                activation=nn.SiLU,
                layer_norm_cls=LayerNorm,
                layer_norm_kw=world_model_config.observation_model.mlp_layer_norm.kw,
            )
            if not config.pixel else None
        )
        observation_model = MultiDecoder(cnn_decoder, mlp_decoder).to(config.device)

        reward_ln_cls = LayerNorm
        reward_model = MLP(
            input_dims=latent_state_size,
            output_dim=world_model_config.reward_model.bins,
            hidden_sizes=[world_model_config.reward_model.dense_units] * world_model_config.reward_model.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": reward_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=reward_ln_cls,
            norm_args={
                **world_model_config.reward_model.layer_norm.kw,
                "normalized_shape": world_model_config.reward_model.dense_units,
            },
        ).to(config.device)

        discount_ln_cls = LayerNorm
        continue_model = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[world_model_config.discount_model.dense_units] * world_model_config.discount_model.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": discount_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=discount_ln_cls,
            norm_args={
                **world_model_config.discount_model.layer_norm.kw,
                "normalized_shape": world_model_config.discount_model.dense_units,
            },
        ).to(config.device)
        world_model = WorldModel(
            encoder.apply(init_weights),
            rssm,
            observation_model.apply(init_weights),
            reward_model.apply(init_weights),
            continue_model.apply(init_weights),
        )

        actor_cls = Actor
        actor: Actor = actor_cls(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            init_std=actor_config.init_std,
            min_std=actor_config.min_std,
            dense_units=actor_config.dense_units,
            activation=nn.SiLU,
            mlp_layers=actor_config.mlp_layers,
            distribution_config=config.distribution,
            layer_norm_cls=LayerNorm,
            layer_norm_kw=actor_config.layer_norm.kw,
            unimix=config.unimix,
            action_clip=actor_config.action_clip,
        ).to(config.device)

        critic_ln_cls = LayerNorm
        critic = MLP(
            input_dims=latent_state_size,
            output_dim=critic_config.bins,
            hidden_sizes=[critic_config.dense_units] * critic_config.mlp_layers,
            activation=nn.SiLU,
            layer_args={"bias": critic_ln_cls == nn.Identity},
            flatten_dim=None,
            norm_layer=critic_ln_cls,
            norm_args={
                **critic_config.layer_norm.kw,
                "normalized_shape": critic_config.dense_units,
            },
        ).to(config.device)
        actor.apply(init_weights)
        critic.apply(init_weights)

        if config.hafner_initialization:
            actor.mlp_heads.apply(uniform_init_weights(1.0))
            critic.model[-1].apply(uniform_init_weights(0.0))
            rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
            rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
            world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
            world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
            if mlp_decoder is not None:
                mlp_decoder.heads.apply(uniform_init_weights(1.0))
            if cnn_decoder is not None:
                cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

        player = PlayerDV3(  # encoder, rssm, actor
            copy.deepcopy(world_model.encoder),
            copy.deepcopy(world_model.rssm),
            copy.deepcopy(actor),
            actions_dim,
            config.parallels,
            config.world_model.stochastic_size,
            config.world_model.recurrent_model.recurrent_state_size,
            config.device,
            discrete_size=config.world_model.discrete_size,
        )
        target_critic = copy.deepcopy(critic)
        # Tie weights between the agent and the player
        for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(world_model.rssm.parameters(), player.rssm.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
            p.data = agent_p.data
        return world_model, actor, critic, target_critic, player
