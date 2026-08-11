from .reinforce import VanillaPolicyGradient
from .actor_critic import (
    ActorCritic,
    SoftActorCritic,
    SoftActorCriticDiscrete,
    PhasicActorCritic,
    DeterministicActorCritic,
    TwinDelayedActorCritic
)
from .deep_q_network import (
    DeepQNetwork,
    DuelingDeepQNetwork,
    NoisyDeepQNetwork,
    C51DeepQNetwork,
    QRDeepQNetwork,
    DeepRecurrentQNetwork
)
from .parameterized_dqn import (
    ParameterizedDQN,
    MultipassParameterizedDQN,
    SplitParameterisedDQN
)
