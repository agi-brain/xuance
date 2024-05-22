import torch
from torch import Tensor
from typing import Tuple
import argparse
from onpolicy.algorithms.actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks
    to compute actions and value function predictions.

    args: (argparse.Namespace)
        Arguments containing relevant model and policy information.
    obs_space: (gym.Space)
        Observation space.
    cent_obs_space: (gym.Space)
        Value function input space
        (centralized input for MAPPO, decentralized for IPPO).
    action_space: (gym.Space) a
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space,
        cent_obs_space,
        act_space,
        device=torch.device("cpu"),
    ) -> None:
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        Decay the actor and critic learning rates.
        episode: (int)
            Current training episode.
        episodes: (int)
            Total number of training episodes.
        """
        update_linear_schedule(
            optimizer=self.actor_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.lr,
        )
        update_linear_schedule(
            optimizer=self.critic_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.critic_lr,
        )

    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions and value function predictions for the given inputs.
        cent_obs (np.ndarray):
            Centralized input to the critic.
        obs (np.ndarray):
            Local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            If actor is RNN, RNN states for actor.
        rnn_states_critic: (np.ndarray)
            If critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            Denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether the action should be mode of
            distribution or should be sampled.

        :return values: (torch.Tensor)
            value function predictions.
        :return actions: (torch.Tensor)
            actions to take.
        :return action_log_probs: (torch.Tensor)
            log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor)
            updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor)
            updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor.forward(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )

        values, rnn_states_critic = self.critic.forward(
            cent_obs, rnn_states_critic, masks
        )
        return (values, actions, action_log_probs, rnn_states_actor, rnn_states_critic)

    def get_values(self, cent_obs, rnn_states_critic, masks) -> Tensor:
        """
        Get value function predictions.
        cent_obs (np.ndarray):
            centralized input to the critic.
        rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic.forward(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action logprobs / entropy and
        value function predictions for actor update.
        cent_obs (np.ndarray):
            centralized input to the critic.
        obs (np.ndarray):
            local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        action: (np.ndarray)
            actions whose log probabilites and entropy to compute.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            denotes whether an agent is active or dead.

        :return values: (torch.Tensor)
            value function predictions.
        :return action_log_probs: (torch.Tensor)
            log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )

        values, _ = self.critic.forward(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions using the given inputs.
        obs (np.ndarray):
            local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            whether the action should be mode of
            distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor.forward(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor
