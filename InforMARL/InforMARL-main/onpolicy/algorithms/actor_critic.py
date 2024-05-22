import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, List
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: gym.Space
        Observation space.
    action_space: gym.Space
        Action space.
    device: torch.device
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self, args, obs_space, action_space, device=torch.device("cpu")
    ) -> None:
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return (actions, action_log_probs, rnn_states)

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return (action_log_probs, dist_entropy)


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")) -> None:
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return (values, rnn_states)
