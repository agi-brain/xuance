from xuance.torch.rl_models.architectures.single_agent import (
    VanillaPolicyGradient,
    ActorCritic,
    SoftActorCritic,
    SoftActorCriticDiscrete,
    PhasicActorCritic,
    DeterministicActorCritic,
    TwinDelayedActorCritic,
    DeepQNetwork,
    DuelingDeepQNetwork,
    NoisyDeepQNetwork,
    C51DeepQNetwork,
    QRDeepQNetwork,
    DeepRecurrentQNetwork,
    ParameterizedDQN,
    MultipassParameterizedDQN,
    SplitParameterisedDQN,
)

from xuance.torch.rl_models.architectures.multi_agent import (
    MixingQNetwork,
    WeightedMixingQNetwork,
    QTranMixingNetwork,
    DeepCoordinationGraph,

    IndependentActorCritic,
    MultiAgentActorCritic,
    CounterfactualMultiAgentActorCritic,
    ValueDecompositionActorCritic,
    IndependentDeterministicActorCritic,
    MultiAgentDeterministicActorCritic,
    IndependentSoftActorCritic,
    MultiAgentSoftActorCritic,
    IndependentTwinDelayedActorCritic,
    MultiAgentTwinDelayedActorCritic,

    MeanFieldQNetwork,
    MeanFiledActorCritic
)

__all__ = [
    "VanillaPolicyGradient",
    "ActorCritic",
    "SoftActorCritic",
    "SoftActorCriticDiscrete",
    "PhasicActorCritic",
    "DeterministicActorCritic",
    "TwinDelayedActorCritic",
    "DeepQNetwork",
    "DuelingDeepQNetwork",
    "NoisyDeepQNetwork",
    "C51DeepQNetwork",
    "QRDeepQNetwork",
    "DeepRecurrentQNetwork",
    "ParameterizedDQN",
    "MultipassParameterizedDQN",
    "SplitParameterisedDQN",

    "MixingQNetwork",
    "WeightedMixingQNetwork",
    "QTranMixingNetwork",
    "DeepCoordinationGraph",

    "IndependentActorCritic",
    "MultiAgentActorCritic",
    "CounterfactualMultiAgentActorCritic",
    "ValueDecompositionActorCritic",
    "IndependentDeterministicActorCritic",
    "MultiAgentDeterministicActorCritic",
    "IndependentSoftActorCritic",
    "MultiAgentSoftActorCritic",
    "IndependentTwinDelayedActorCritic",
    "MultiAgentTwinDelayedActorCritic",

    "MeanFieldQNetwork",
    "MeanFiledActorCritic"
]
