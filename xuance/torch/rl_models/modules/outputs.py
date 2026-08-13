import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from xuance.torch import Tensor


@dataclass
class RNN_State:
    hidden_states: Optional[Tensor] | None = None
    cell_states: Optional[Tensor] | None = None


@dataclass
class RepresentationOutput:
    embeddings: Tensor
    rnn_states: RNN_State | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class StochasticActorOutput:
    representations: RepresentationOutput
    distributions: Optional[torch.distributions] = None


@dataclass
class DeterministicActorOutput:
    representations: RepresentationOutput
    actions: Optional[Tensor] | None = None


@dataclass
class CriticOutput:
    representations: RepresentationOutput | dict
    values: Dict[str, Tensor] | Tensor


@dataclass
class TwinCriticOutput:
    representations_1: RepresentationOutput
    representations_2: RepresentationOutput
    values_1: Tensor
    values_2: Tensor


@dataclass
class ModelOutput:
    actions: Optional[Tensor] | None = None
    distributions: Optional[torch.distributions] = None
    values: Optional[Tensor] | None = None
    rep_out: RepresentationOutput | None = None
    actor_rep_out: RepresentationOutput | None = None
    critic_rep_out: RepresentationOutput | None = None


@dataclass
class MultiAgentModelOutput:
    actions: Dict[str, Tensor] | None = None
    log_probs: Dict[str, Tensor] | None = None
    distributions: Dict[str, torch.distributions] | None = None
    values: Dict[str, Tensor] | None = None
    rnn_states: Dict[str, RNN_State] | None = None
    actor_rnn_states: Dict[str, RNN_State] | None = None
    critic_rnn_states: Dict[str, RNN_State] | None = None
    rep_out: Dict[str, RepresentationOutput] | None = None
    actor_rep_out: Dict[str, RepresentationOutput] | None = None
    critic_rep_out: Dict[str, RepresentationOutput] | None = None


@dataclass
class DRL_Batch:
    batch_size: int
    observations: Tensor
    actions: Tensor
    next_observations: Optional[Tensor] | None = None
    rewards: Optional[Tensor] | None = None
    terminals: Optional[Tensor] | None = None
    values: Optional[Tensor] | None = None
    returns: Optional[Tensor] | None = None
    advantages: Optional[Tensor] | None = None
    avail_actions: Optional[Tensor] | None = None
    old_log_probs: Optional[Tensor] | None = None
    old_distributions: Optional[Any] = None


@dataclass
class MARLBatch:
    batch_size: int
    seq_length: int
    agent_indices: Dict[str, Tensor]
    observations: Dict[str, Tensor] | Any
    actions: Dict[str, Tensor] | Any
    global_states: Optional[Tensor] | None = None
    avail_actions: Dict[str, Tensor] | None = None
    agent_masks: Dict[str, Tensor] | None = None
    filled_masks: Tensor | None = None  # Shape: batch_size * seq_length

    def valid_mask(
            self,
            group: str,  # The selected group key
            n_agents: int  # Number of agents in the group
    ) -> Tensor:
        mask = self.agent_masks[group]

        if self.filled_masks is not None:
            mask = mask * self.filled_masks.unsqueeze(1).repeat(1, n_agents, 1).reshape([-1, self.seq_length])

        return mask

    def flatten_valid(self, x, group):
        return x.reshape(-1), self.valid_mask(group).reshape(-1)


@dataclass
class OnPolicyMARLBatch(MARLBatch):
    returns: Dict[str, Tensor] | None = None
    values: Dict[str, Tensor] | None = None
    advantages: Dict[str, Tensor] | None = None
    old_log_probs: Dict[str, Tensor] | None = None


@dataclass
class OffPolicyMARLBatch(MARLBatch):
    next_observations: Dict[str, Tensor] | None = None
    next_global_states: Tensor | None = None
    next_avail_actions: Dict[str, Tensor] | None = None
    rewards: Dict[str, Tensor] | None = None
    terminals: Dict[str, Tensor] | None = None

