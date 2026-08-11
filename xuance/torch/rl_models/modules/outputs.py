import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from xuance.torch import Tensor


@dataclass
class RNN_State:
    hidden_states: Optional[Tensor] = None
    cell_states: Optional[Tensor] = None


@dataclass
class RepresentationOutput:
    embeddings: Tensor
    rnn_states: Optional[RNN_State] = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class StochasticActorOutput:
    representations: RepresentationOutput
    distributions: Optional[torch.distributions] = None


@dataclass
class DeterministicActorOutput:
    representations: RepresentationOutput
    actions: Optional[Tensor] = None


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
    actions: Optional[Tensor] = None
    distributions: Optional[torch.distributions] = None
    values: Optional[Tensor] = None
    rep_out: Optional[RepresentationOutput] = None
    actor_rep_out: Optional[RepresentationOutput] = None
    critic_rep_out: Optional[RepresentationOutput] = None


@dataclass
class MultiAgentModelOutput:
    actions: Optional[Dict[str, Tensor]] = None
    log_probs: Optional[Dict[str, Tensor]] = None
    distributions: Optional[Dict[str, torch.distributions]] = None
    values: Optional[Dict[str, Tensor]] = None
    rnn_states: Optional[Dict[str, RNN_State]] = None
    actor_rnn_states: Optional[Dict[str, RNN_State]] = None
    critic_rnn_states: Optional[Dict[str, RNN_State]] = None
    rep_out: Optional[Dict[str, RepresentationOutput]] = None
    actor_rep_out: Optional[Dict[str, RepresentationOutput]] = None
    critic_rep_out: Optional[Dict[str, RepresentationOutput]] = None
