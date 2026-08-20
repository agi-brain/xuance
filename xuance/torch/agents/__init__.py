from .base import Agent, MARLAgents, RandomAgents
from .core import OnPolicyAgent, OffPolicyAgent, OffPolicyMARLAgents, OnPolicyMARLAgents, OfflineAgent

'''Single-Agent Reinforcement Learning algorithms'''
from .policy_gradient import (
    # on-policy
    PG_Agent,
    A2C_Agent,
    PPO_Agent,
    PPOKL_Agent,
    PPG_Agent,
    NPG_Agent,
    # off-policy
    DDPG_Agent,
    TD3_Agent,
    PDQN_Agent,
    MPDQN_Agent,
    SPDQN_Agent,
    SAC_Agent,
)
from .qlearning_family import (
    DQN_Agent,
    DuelDQN_Agent,
    DDQN_Agent,
    NoisyDQN_Agent,
    C51_Agent,
    QRDQN_Agent,
    PerDQN_Agent,
    DRQN_Agent
)

'''Model-based Reinforcement Learning'''
from .model_based_rl import DreamerV2Agent, DreamerV3Agent

'''Multi-Agent Reinforcement Learning Algorithms'''
from .multi_agent_rl import (
    # value-based MARL
    IQL_Agents,
    VDN_Agents,
    QMIX_Agents,
    WQMIX_Agents,
    QTRAN_Agents,
    DCG_Agents,
    MFQ_Agents,
    # on-policy-based MARL
    IAC_Agents,
    VDAC_Agents,
    COMA_Agents,
    IPPO_Agents,
    MAPPO_Agents,
    MFAC_Agents,
    # off-policy actor-critic-based MARL
    IDDPG_Agents,
    MADDPG_Agents,
    ITD3_Agents,
    MATD3_Agents,
    ISAC_Agents,
    MASAC_Agents,
    # communication-based MARL
    CommNet_Agents,
    IC3Net_Agents,
    TarMAC_Agents
)

from .offline_rl import (
    TD3_BC_Agent
)

from .contrastive_unsupervised_rl import (
    CURL_Agent,
    SPR_Agent,
    DrQ_Agent
)

REGISTRY_Agents = {
    "PG": PG_Agent,
    "A2C": A2C_Agent,
    "PPO": PPO_Agent,
    "PPO_KL": PPOKL_Agent,
    "PPG": PPG_Agent,
    "DDPG": DDPG_Agent,
    "SAC": SAC_Agent,
    "TD3": TD3_Agent,
    "DQN": DQN_Agent,
    "Duel_DQN": DuelDQN_Agent,
    "DDQN": DDQN_Agent,
    "NoisyDQN": NoisyDQN_Agent,
    "PerDQN": PerDQN_Agent,
    "C51DQN": C51_Agent,
    "QRDQN": QRDQN_Agent,
    "PDQN": PDQN_Agent,
    "MPDQN": MPDQN_Agent,
    "SPDQN": SPDQN_Agent,
    "DRQN": DRQN_Agent,
    "NPG": NPG_Agent,

    "DreamerV2": DreamerV2Agent,
    "DreamerV3": DreamerV3Agent,

    "RANDOM": RandomAgents,
    "IQL": IQL_Agents,
    "VDN": VDN_Agents,
    "QMIX": QMIX_Agents,
    "CWQMIX": WQMIX_Agents,
    "OWQMIX": WQMIX_Agents,
    "QTRAN_base": QTRAN_Agents,
    "QTRAN_alt": QTRAN_Agents,
    "DCG": DCG_Agents,
    "DCG_S": DCG_Agents,
    "IAC": IAC_Agents,
    "VDAC": VDAC_Agents,
    "COMA": COMA_Agents,
    "IDDPG": IDDPG_Agents,
    "MADDPG": MADDPG_Agents,
    "MFQ": MFQ_Agents,
    "MFAC": MFAC_Agents,
    "IPPO": IPPO_Agents,
    "MAPPO": MAPPO_Agents,
    "ISAC": ISAC_Agents,
    "MASAC": MASAC_Agents,
    "ITD3": ITD3_Agents,
    "MATD3": MATD3_Agents,
    "IC3Net": IC3Net_Agents,
    "CommNet": CommNet_Agents,
    "TarMAC": TarMAC_Agents,

    "TD3BC": TD3_BC_Agent,

    "CURL": CURL_Agent,
    "SPR": SPR_Agent,
    "DrQ": DrQ_Agent,
}

__all__ = [
    "Agent",
    "MARLAgents",
    "RandomAgents",

    "OnPolicyAgent",
    "OffPolicyAgent",
    "OffPolicyMARLAgents",
    "OnPolicyMARLAgents",
    "OfflineAgent",

    "REGISTRY_Agents",

    "PG_Agent",
    "A2C_Agent",
    "PPO_Agent",
    "PPOKL_Agent",
    "PPG_Agent",
    "DDPG_Agent",
    "TD3_Agent",
    "PDQN_Agent",
    "MPDQN_Agent",
    "SPDQN_Agent",
    "SAC_Agent",
    "DQN_Agent",
    "DuelDQN_Agent",
    "DDQN_Agent",
    "NoisyDQN_Agent",
    "C51_Agent",
    "QRDQN_Agent",
    "PerDQN_Agent",
    "DRQN_Agent",
    "NPG_Agent",

    "DreamerV2Agent",
    "DreamerV3Agent",

    "IQL_Agents",
    "VDN_Agents",
    "QMIX_Agents",
    "WQMIX_Agents",
    "QTRAN_Agents",
    "DCG_Agents",
    "IAC_Agents",
    "VDAC_Agents",
    "COMA_Agents",
    "IDDPG_Agents",
    "MADDPG_Agents",
    "IC3Net_Agents",
    "CommNet_Agents",
    "TarMAC_Agents",
    "MFQ_Agents",
    "MFAC_Agents",
    "IPPO_Agents",
    "MAPPO_Agents",
    "ISAC_Agents",
    "MASAC_Agents",
    "MATD3_Agents",

    "TD3_BC_Agent",

    "CURL_Agent",
    "SPR_Agent",
    "DrQ_Agent"
]
