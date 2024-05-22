"""
Modules to compute actions. Basically the final layer in the network.
This imports modified probability distribution layers wherein we can give action
masks to re-normalise the probability distributions
"""
from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
from typing import Optional


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    action_space: (gym.Space) action space.
    inputs_dim: int
        Dimension of network input.
    use_orthogonal: bool
        Whether to use orthogonal weight init or xavier uniform.
    gain: float
        Gain of the output layer of the network.
    """

    def __init__(
        self, action_space, inputs_dim: int, use_orthogonal: bool, gain: float
    ):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(
                    Categorical(inputs_dim, action_dim, use_orthogonal, gain)
                )
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList(
                [
                    DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain),
                    Categorical(inputs_dim, discrete_dim, use_orthogonal, gain),
                ]
            )

    def forward(
        self,
        x: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        deterministic: bool = False,
    ):
        """
        Compute actions and action logprobs from given input.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: bool
            Whether to sample from action distribution or return the mode.

        :return actions: torch.Tensor
            actions to take.
        :return action_log_probs: torch.Tensor
            log probabilities of taken actions.
        """
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True
            )

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def get_probs(
        self, x: torch.Tensor, available_actions: Optional[torch.tensor] = None
    ):
        """
        Compute action probabilities from inputs.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)

        :return action_probs: torch.Tensor
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(
        self,
        x: torch.tensor,
        action: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        active_masks: Optional[torch.tensor] = None,
    ):
        """
        Compute log probability and entropy of given actions.
        x: torch.Tensor
            Input to network.
        action: torch.Tensor
            Actions whose entropy and log probability to evaluate.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: torch.Tensor
            Denotes whether an agent is active or dead.

        :return action_log_probs: torch.Tensor
            log probabilities of the input actions.
        :return dist_entropy: torch.Tensor
            action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks).sum()
                            / active_masks.sum()
                        )
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum()
                            / active_masks.sum()
                        )
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True
            )
            dist_entropy = (
                dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98
            )  #! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append(
                        (action_logit.entropy() * active_masks.squeeze(-1)).sum()
                        / active_masks.sum()
                    )
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)  # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (
                    action_logits.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
