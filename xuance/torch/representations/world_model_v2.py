import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from xuance.torch.utils.distributions import TruncatedNormal
from xuance.torch.utils.layers4dreamder import (
    CNN,
    MLP,
    DeCNN,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
    ModuleType,
    cnn_forward
)
from xuance.torch.utils import dotdict, init_weights, compute_stochastic_state

# diff(v3: layer_norm_cls)
# checked
class CNNEncoder(nn.Module):
    """The Dreamer-V2 image encoder. This is composed of 4 `nn.Conv2d` with
    kernel_size=3, stride=2 and padding=1. No bias is used if a `nn.LayerNorm`
    is used after the convolution. This 4-stages model assumes that the image
    is a 64x64. If more than one image is to be encoded, then those will
    be concatenated on the channel dimension and fed to the encoder.

    Args:
        keys (Sequence[str]): the keys representing the image observations to encode.
        input_channels (Sequence[int]): the input channels, one for each image observation to encode.
        image_size (Tuple[int, int]): the image size as (Height,Width).
        channels_multiplier (int): the multiplier for the output channels. Given the 4 stages, the 4 output channels
            will be [1, 2, 4, 8] * `channels_multiplier`.
        layer_norm (bool, optional): whether to apply the layer normalization.
            Defaults to True.
        activation (ModuleType, optional): the activation function.
            Defaults to nn.ELU.
    """

    def __init__(
            self,
            input_channels: Sequence[int],
            image_size: Tuple[int, int],
            channels_multiplier: int,
            layer_norm: bool = False,
            activation: ModuleType = nn.ELU,
    ) -> None:
        super().__init__()
        self.input_dim = (sum(input_channels), *image_size)
        self.model = nn.Sequential(
            CNN(
                input_channels=sum(input_channels),
                hidden_channels=(torch.tensor([1, 2, 4, 8]) * channels_multiplier).tolist(),
                layer_args={"kernel_size": 4, "stride": 2},
                activation=activation,
                norm_layer=[LayerNormChannelLast for _ in range(4)] if layer_norm else None,
                norm_args=(
                    [{"normalized_shape": (2 ** i) * channels_multiplier} for i in range(4)] if layer_norm else None
                ),
            ),
            nn.Flatten(-3, -1),
        )
        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]

    def forward(self, obs: Tensor) -> Tensor:
        # x = torch.cat([obs[k] for k in self.keys], -3)  # channels dimension
        return cnn_forward(self.model, obs, obs.shape[-3:], (-1,))

# diff(v3: layer_norm_cls)
# checked
class CNNDecoder(nn.Module):
    """The almost-exact inverse of the `CNNEncoder` class, where in 4 stages it reconstructs
    the observation image to 64x64. If multiple images are to be reconstructed,
    then it will create a dictionary with an entry for every reconstructed image.
    No bias is used if a `nn.LayerNorm` is used after the `nn.Conv2dTranspose` layer.

    Args:
        keys (Sequence[str]): the keys of the image observation to be reconstructed.
        output_channels (Sequence[int]): the output channels, one for every image observation.
        channels_multiplier (int): the channels multiplier, same for the encoder network.
        latent_state_size (int): the size of the latent state. Before applying the decoder,
            a `nn.Linear` layer is used to project the latent state to a feature vector.
        cnn_encoder_output_dim (int): the output of the image encoder.
        image_size (Tuple[int, int]): the final image size.
        activation (nn.Module, optional): the activation function.
            Defaults to nn.ELU.
        layer_norm (bool, optional): whether to apply the layer normalization.
            Defaults to True.
    """

    def __init__(
            self,
            output_channels: Sequence[int],
            channels_multiplier: int,
            latent_state_size: int,
            cnn_encoder_output_dim: int,
            image_size: Tuple[int, int],
            activation: nn.Module = nn.ELU,
            layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.cnn_encoder_output_dim = cnn_encoder_output_dim
        self.image_size = image_size
        self.output_dim = (sum(output_channels), *image_size)
        self.model = nn.Sequential(
            nn.Linear(latent_state_size, cnn_encoder_output_dim),
            nn.Unflatten(1, (cnn_encoder_output_dim, 1, 1)),
            DeCNN(
                input_channels=cnn_encoder_output_dim,
                hidden_channels=(torch.tensor([4, 2, 1]) * channels_multiplier).tolist() + [self.output_dim[0]],
                layer_args=[
                    {"kernel_size": 5, "stride": 2},
                    {"kernel_size": 5, "stride": 2},
                    {"kernel_size": 6, "stride": 2},
                    {"kernel_size": 6, "stride": 2},
                ],
                activation=[activation, activation, activation, None],
                norm_layer=[LayerNormChannelLast for _ in range(3)] + [None] if layer_norm else None,
                norm_args=(
                    [{"normalized_shape": (2 ** (4 - i - 2)) * channels_multiplier} for i in range(self.output_dim[0])]
                    + [None]
                    if layer_norm
                    else None
                ),
            ),
        )

    def forward(self, latent_states: Tensor) -> List[Tensor]:
        cnn_out = cnn_forward(self.model, latent_states, (latent_states.shape[-1],), self.output_dim)
        return torch.split(cnn_out, self.output_channels, -3)

# diff(v3: layer_norm_cls, sym_log)
# checked
class MLPEncoder(nn.Module):
    """The Dreamer-V3 vector encoder. This is composed of N `nn.Linear` layers, where
    N is specified by `mlp_layers`. No bias is used if a `nn.LayerNorm` is used after the linear layer.
    If more than one vector is to be encoded, then those will concatenated on the last
    dimension before being fed to the encoder.

    Args:
        keys (Sequence[str]): the keys representing the vector observations to encode.
        input_dims (Sequence[int]): the dimensions of every vector to encode.
        mlp_layers (int, optional): how many mlp layers.
            Defaults to 4.
        dense_units (int, optional): the dimension of every mlp.
            Defaults to 512.
        layer_norm (bool, optional): whether to apply the layer normalization.
            Defaults to True.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.ELU.
    """

    def __init__(
            self,
            input_dims: Sequence[int],
            mlp_layers: int = 4,
            dense_units: int = 512,
            layer_norm: bool = False,
            activation: ModuleType = nn.ELU,
    ) -> None:
        super().__init__()
        self.input_dim = sum(input_dims)
        self.model = MLP(
            self.input_dim,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )
        self.output_dim = dense_units

    def forward(self, obs: Tensor) -> Tensor:
        # x = torch.cat([obs[k] for k in self.keys], -1)
        return self.model(obs)

# diff(v3: layer_norm_cls)
# checked
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
        layer_norm (bool, optional): whether to apply the layer normalization.
            Defaults to True.
        activation (ModuleType, optional): the activation function after every layer.
            Defaults to nn.ELU.
    """

    def __init__(
            self,
            output_dims: Sequence[int],
            latent_state_size: int,
            mlp_layers: int = 4,
            dense_units: int = 512,
            activation: ModuleType = nn.ELU,
            layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.output_dims = output_dims
        self.model = MLP(
            latent_state_size,
            None,
            [dense_units] * mlp_layers,
            activation=activation,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )
        self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])

    def forward(self, latent_states: Tensor) -> Dict[str, Tensor]:
        x = self.model(latent_states)
        return self.heads[0](x)  # revised to adapt to xuance


# diff(v2: gru, bias True; v3: layer_norm_cls)
# checked
class RecurrentModel(nn.Module):
    """Recurrent model for the model-base Dreamer-V3 agent.
    This implementation uses the `models.LayerNormGRUCell`, which combines
    the standard GRUCell from PyTorch with the `nn.LayerNorm`, where the normalization is applied
    right after having computed the projection from the input to the weight space.

    Args:
        input_size (int): the input size of the model.
        recurrent_state_size (int): the size of the recurrent state.
        dense_units (int): the number of dense units.
        activation (nn.Module): the activation function.
            Default to ELU.
        layer_norm (bool): whether to use the LayerNorm inside the GRU.
            Defaults to True.
    """

    def __init__(
            self,
            input_size: int,
            recurrent_state_size: int,
            dense_units: int,
            activation: nn.Module = nn.ELU,
            layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation,
            norm_layer=[nn.LayerNorm] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units}] if layer_norm else None,
        )
        self.rnn = LayerNormGRUCell(
            dense_units, recurrent_state_size, bias=True, batch_first=False, layer_norm_cls=nn.LayerNorm
        )

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


# diff(v2: no unimix, no learnable_initial_state)
class RSSM(nn.Module):
    """RSSM model for the model-base Dreamer agent.

    Args:
        recurrent_model (nn.Module): the recurrent model of the RSSM model described in
        [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        representation_model (nn.Module): the representation model composed by a multi-layer perceptron
            to compute the stochastic part of the latent state.
            For more information see [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        transition_model (nn.Module): the transition model described in
            [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
            The model is composed by a multu-layer perceptron to predict the stochastic part of the latent state.
        distribution_config (Dict[str, Any]): the configs of the distributions.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
    """

    def __init__(
            self,
            recurrent_model: nn.Module,
            representation_model: nn.Module,
            transition_model: nn.Module,
            distribution_config: Dict[str, Any],
            discrete: Optional[int] = 32,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.discrete = discrete
        self.distribution_config = distribution_config

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
        posterior = (1 - is_first) * posterior.view(*posterior.shape[:-2], -1)
        recurrent_state = (1 - is_first) * recurrent_state
        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

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
        logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

    def _transition(self, recurrent_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """
        logits = self.transition_model(recurrent_out)
        return logits, compute_stochastic_state(logits, discrete=self.discrete)

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

# diff(v2: continuous, trunc_normal_dist, no max_std; v3: scaled_normal_dist)
# checked
class Actor(nn.Module):
    """
    The wrapper class of the Dreamer_v2 Actor model.

    Args:
        latent_state_size (int): the dimension of the latent state (stochastic size + recurrent_state_size).
        actions_dim (Sequence[int]): the dimension in output of the actor.
            The number of actions if continuous, the dimension of the action if discrete.
        is_continuous (bool): whether or not the actions are continuous.
        distribution_config (Dict[str, Any]): The configs of the distributions.
        init_std (float): the amount to sum to the input of the softplus function for the standard deviation.
            Default to 5.
        min_std (float): the minimum standard deviation for the actions.
            Default to 0.1.
        dense_units (int): the dimension of the hidden dense layers.
            Default to 400.
        activation (int): the activation function to apply after the dense layers.
            Default to nn.ELU.
        mlp_layers (int): the number of linear layers.
            Default to 4.
        layer_norm (bool): whether or not to use the layer norm.
            Default to False.
        expl_amount (float): the exploration amount to use during training.
            Default to 0.0.
        expl_decay (float): the exploration decay to use during training.
            Default to 0.0.
        expl_min (float): the exploration amount minimum to use during training.
            Default to 0.0.
    """

    def __init__(
            self,
            latent_state_size: int,
            actions_dim: Sequence[int],
            is_continuous: bool,
            distribution_config: Dict[str, Any],
            init_std: float = 0.0,
            min_std: float = 0.1,
            dense_units: int = 400,
            activation: nn.Module = nn.ELU,
            mlp_layers: int = 4,
            layer_norm: bool = False,
            expl_amount: float = 0.0,
            expl_decay: float = 0.0,
            expl_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.distribution_config = distribution_config
        self.distribution = distribution_config.get("type", "auto").lower()
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete", "trunc_normal"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal`, `tanh_normal` and `trunc_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous:
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        if self.distribution == "auto":
            if is_continuous:
                self.distribution = "trunc_normal"
            else:
                self.distribution = "discrete"
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )
        if is_continuous:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, sum(actions_dim) * 2)])
        else:
            self.mlp_heads = nn.ModuleList([nn.Linear(dense_units, action_dim) for action_dim in actions_dim])
        self.actions_dim = actions_dim
        self.is_continuous = is_continuous
        self.init_std = torch.tensor(init_std)  # 0.0
        self.min_std = min_std  # 0.1
        self.distribution_config = distribution_config
        self._expl_amount = expl_amount
        self._expl_decay = expl_decay
        self._expl_min = expl_min

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
            mask (Dict[str, Tensor], optional): the action mask (which actions can be selected).
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
            elif self.distribution == "trunc_normal":
                # std: [0.1, 2.1] ??
                std = 2 * torch.sigmoid((std + self.init_std) / 2) + self.min_std
                dist = TruncatedNormal(torch.tanh(mean), std, -1, 1)
                actions_dist = Independent(dist, 1)
            else:
                actions_dist = None
            if not greedy:
                actions = actions_dist.rsample()
            else:
                actions = actions_dist.mean  # TruncatedNormal'> does not implement mode
            actions = [actions]
            actions_dist = [actions_dist]
        else:
            actions_dist = []
            actions = []
            for logits in pre_dist:
                actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
                if not greedy:
                    actions.append(actions_dist[-1].rsample())
                else:
                    actions.append(actions_dist[-1].mode)
        return tuple(actions), tuple(actions_dist)


# diff(v3: rssm; v2: recurrent_model & representation)
# checked
class PlayerDV2(nn.Module):
    """
    The model of the Dreamer_v2 player.

    Args:
        encoder (nn.Module): the encoder.
        recurrent_model (nn.Module): the recurrent model.
        representation_model (nn.Module): the representation model.
        actor (nn.Module): the actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (str | torch.device): the device where the model is stored.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
            Defaults to 32.
        actor_type (str, optional): which actor the player is using ('task' or 'exploration').
            Default to None.
    """

    def __init__(
            self,
            encoder: MultiEncoder,
            recurrent_model: nn.Module,
            representation_model: nn.Module,
            actor: nn.Module,
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
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size
        self.actor_type = actor_type

        self.actions, self.recurrent_state, self.stochastic_state = [None] * 3

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
            self.recurrent_state = torch.zeros(1, num_envs, self.recurrent_state_size, device=self.device)
            self.stochastic_state = torch.zeros(
                1, num_envs, self.stochastic_size * self.discrete_size, device=self.device
            )
        else:
            self.actions[:, reset_envs] = torch.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs] = torch.zeros_like(self.recurrent_state[:, reset_envs])
            self.stochastic_state[:, reset_envs] = torch.zeros_like(self.stochastic_state[:, reset_envs])

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
            mask (Dict[str, Tensor], optional): the action mask (which actions can be selected).
                Default to None.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        posterior_logits = self.representation_model(torch.cat((self.recurrent_state, embedded_obs), -1))
        stochastic_state = compute_stochastic_state(posterior_logits, discrete=self.discrete_size)
        self.stochastic_state = stochastic_state.view(
            *stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        actions, _ = self.actor(torch.cat((self.stochastic_state, self.recurrent_state), -1), greedy, mask)
        self.actions = torch.cat(actions, -1)
        return actions


# same
# checked
class WorldModel(nn.Module):
    """
    Wrapper class for the World model.

    Args:
        encoder (nn.Module): the encoder.
        rssm (RSSM): the rssm.
        observation_model (nn.Module): the observation model.
        reward_model (nn.Module): the reward model.
        continue_model (nn.Module, optional): the continue model.
    """

    def __init__(
            self,
            encoder: MultiEncoder,
            rssm: RSSM,
            observation_model,
            reward_model,
            continue_model: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.encoder: MultiEncoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model


class DreamerV2WorldModel(nn.Module):
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
            self.obs_space = gym.spaces.Box(0, 255, ((self.obs_space.shape[2],) + self.obs_space.shape[:2]), np.uint8)
        self.world_model, self.actor, self.critic, self.target_critic, self.player = (
            DreamerV2WorldModel._build_model(
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

        world_model_state: Optional[Dict[str, Tensor]] = None,
        actor_state: Optional[Dict[str, Tensor]] = None,
        critic_state: Optional[Dict[str, Tensor]] = None,
        target_critic_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[WorldModel, Actor, nn.Module, nn.Module, PlayerDV2]:
        """Build the models and wrap them with Fabric.

        Args:
            actions_dim (Sequence[int]): the dimension of the actions.
            is_continuous (bool): whether or not the actions are continuous.
            config (DictConfig): the configs.
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

        # Sizes
        stochastic_size = world_model_config.stochastic_size * world_model_config.discrete_size
        latent_state_size = stochastic_size + world_model_config.recurrent_model.recurrent_state_size

        # Define models
        cnn_encoder = (
            CNNEncoder(
                input_channels=[int(np.prod(obs_space.shape[:-2]))],
                image_size=obs_space.shape[-2:],
                channels_multiplier=world_model_config.encoder.cnn_channels_multiplier,
                layer_norm=world_model_config.encoder.layer_norm,
                activation=nn.ELU,
            )
            if config.pixel else None
        )
        mlp_encoder = (
            MLPEncoder(
                input_dims=[obs_space.shape[0]],
                mlp_layers=world_model_config.encoder.mlp_layers,
                dense_units=world_model_config.encoder.dense_units,
                activation=nn.ELU,
                layer_norm=world_model_config.encoder.layer_norm,
            )
            if not config.pixel else None
        )
        encoder: MultiEncoder = MultiEncoder(cnn_encoder, mlp_encoder).to(config.device)

        recurrent_model = RecurrentModel(
            **world_model_config.recurrent_model,
            input_size=int(sum(actions_dim) + stochastic_size),
        )
        representation_model = MLP(
            input_dims=(
                    world_model_config.recurrent_model.recurrent_state_size +
                    encoder.cnn_output_dim +
                    encoder.mlp_output_dim
            ),
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.representation_model.hidden_size],
            activation=nn.ELU,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm] if world_model_config.representation_model.layer_norm else None,
            norm_args=(
                [{"normalized_shape": world_model_config.representation_model.hidden_size}]
                if world_model_config.representation_model.layer_norm
                else None
            ),
        )
        transition_model = MLP(
            input_dims=world_model_config.recurrent_model.recurrent_state_size,
            output_dim=stochastic_size,
            hidden_sizes=[world_model_config.transition_model.hidden_size],
            activation=nn.ELU,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm] if world_model_config.transition_model.layer_norm else None,
            norm_args=(
                [{"normalized_shape": world_model_config.transition_model.hidden_size}]
                if world_model_config.transition_model.layer_norm
                else None
            ),
        )
        rssm = RSSM(
            recurrent_model=recurrent_model.apply(init_weights),
            representation_model=representation_model.apply(init_weights),
            transition_model=transition_model.apply(init_weights),
            distribution_config=config.distribution,
            discrete=world_model_config.discrete_size,
        ).to(config.device)
        cnn_decoder = (
            CNNDecoder(
                output_channels=[int(np.prod(obs_space.shape[:-2]))],
                channels_multiplier=world_model_config.observation_model.cnn_channels_multiplier,
                latent_state_size=latent_state_size,
                cnn_encoder_output_dim=cnn_encoder.output_dim,
                image_size=obs_space.shape[-2:],
                activation=nn.ELU,
                layer_norm=world_model_config.observation_model.layer_norm,
            )
            if config.pixel else None
        )
        mlp_decoder = (
            MLPDecoder(
                output_dims=[obs_space.shape[0]],
                latent_state_size=latent_state_size,
                mlp_layers=world_model_config.observation_model.mlp_layers,
                dense_units=world_model_config.observation_model.dense_units,
                activation=nn.ELU,
                layer_norm=world_model_config.observation_model.layer_norm,
            )
            if not config.pixel else None
        )
        observation_model = MultiDecoder(cnn_decoder, mlp_decoder).to(config.device)
        reward_model = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[world_model_config.reward_model.dense_units] * world_model_config.reward_model.mlp_layers,
            activation=nn.ELU,
            flatten_dim=None,
            norm_layer=(
                [nn.LayerNorm for _ in range(world_model_config.reward_model.mlp_layers)]
                if world_model_config.reward_model.layer_norm
                else None
            ),
            norm_args=(
                [
                    {"normalized_shape": world_model_config.reward_model.dense_units}
                    for _ in range(world_model_config.reward_model.mlp_layers)
                ]
                if world_model_config.reward_model.layer_norm
                else None
            ),
        ).to(config.device)
        if world_model_config.use_continues:
            continue_model = MLP(
                input_dims=latent_state_size,
                output_dim=1,
                hidden_sizes=[world_model_config.discount_model.dense_units] * world_model_config.discount_model.mlp_layers,
                activation=nn.ELU,
                flatten_dim=None,
                norm_layer=(
                    [nn.LayerNorm for _ in range(world_model_config.discount_model.mlp_layers)]
                    if world_model_config.discount_model.layer_norm
                    else None
                ),
                norm_args=(
                    [
                        {"normalized_shape": world_model_config.discount_model.dense_units}
                        for _ in range(world_model_config.discount_model.mlp_layers)
                    ]
                    if world_model_config.discount_model.layer_norm
                    else None
                ),
            ).to(config.device)
        else:
            continue_model = None
        world_model = WorldModel(
            encoder.apply(init_weights),
            rssm,
            observation_model.apply(init_weights),
            reward_model.apply(init_weights),
            continue_model.apply(init_weights) if world_model_config.use_continues else None,
        )
        actor_cls = Actor
        actor: Actor = actor_cls(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            init_std=actor_config.init_std,
            min_std=actor_config.min_std,
            mlp_layers=actor_config.mlp_layers,
            dense_units=actor_config.dense_units,
            activation=nn.ELU,
            distribution_config=config.distribution,
            layer_norm=actor_config.layer_norm,
        ).to(config.device)
        critic = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[critic_config.dense_units] * critic_config.mlp_layers,
            activation=nn.ELU,
            flatten_dim=None,
            norm_layer=[nn.LayerNorm for _ in range(critic_config.mlp_layers)] if critic_config.layer_norm else None,
            norm_args=(
                [{"normalized_shape": critic_config.dense_units} for _ in range(critic_config.mlp_layers)]
                if critic_config.layer_norm
                else None
            ),
        ).to(config.device)
        actor.apply(init_weights)
        critic.apply(init_weights)

        # encoder, recurrent, representation, actor
        player = PlayerDV2(
            copy.deepcopy(world_model.encoder),
            copy.deepcopy(world_model.rssm.recurrent_model),
            copy.deepcopy(world_model.rssm.representation_model),
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
        for agent_p, p in zip(world_model.rssm.recurrent_model.parameters(), player.recurrent_model.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(world_model.rssm.representation_model.parameters(), player.representation_model.parameters()):
            p.data = agent_p.data
        for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
            p.data = agent_p.data
        return world_model, actor, critic, target_critic, player
