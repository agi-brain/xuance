from .base import Agent, MARLAgents, RandomAgents
from .core import OnPolicyAgent, OffPolicyAgent, OffPolicyMARLAgents, OnPolicyMARLAgents

'''Single-Agent Reinforcement Learning algorithms'''
from .policy_gradient import PG_Agent
from .policy_gradient import A2C_Agent
from .policy_gradient import PPOCLIP_Agent
from .policy_gradient import PPOKL_Agent
from .policy_gradient import PPG_Agent
from .policy_gradient import DDPG_Agent
from .policy_gradient import TD3_Agent
from .policy_gradient import PDQN_Agent
from .policy_gradient import MPDQN_Agent
from .policy_gradient import SPDQN_Agent
from .policy_gradient import SAC_Agent
from .policy_gradient import SACDIS_Agent
from .qlearning_family import DQN_Agent
from .qlearning_family import DuelDQN_Agent
from .qlearning_family import DDQN_Agent
from .qlearning_family import NoisyDQN_Agent
from .qlearning_family import C51_Agent
from .qlearning_family import QRDQN_Agent
from .qlearning_family import PerDQN_Agent
from .qlearning_family import DRQN_Agent

'''Multi-Agent Reinforcement Learning Algorithms'''
from .multi_agent_rl import IQL_Agents
from .multi_agent_rl import VDN_Agents
from .multi_agent_rl import QMIX_Agents
from .multi_agent_rl import WQMIX_Agents
from .multi_agent_rl import QTRAN_Agents
from .multi_agent_rl import DCG_Agents
from .multi_agent_rl import VDAC_Agents
from .multi_agent_rl import COMA_Agents
from .multi_agent_rl import IDDPG_Agents
from .multi_agent_rl import MADDPG_Agents
from .multi_agent_rl import MFQ_Agents
from .multi_agent_rl import MFAC_Agents
from .multi_agent_rl import IPPO_Agents
from .multi_agent_rl import MAPPO_Agents
from .multi_agent_rl import ISAC_Agents
from .multi_agent_rl import MASAC_Agents
from .multi_agent_rl import MATD3_Agents

REGISTRY_Agents = {
    "PG": PG_Agent,
    "A2C": A2C_Agent,
    "PPO_Clip": PPOCLIP_Agent,
    "PPO_KL": PPOKL_Agent,
    "PPG": PPG_Agent,
    "DDPG": DDPG_Agent,
    "SAC": SAC_Agent,
    "SACDIS": SACDIS_Agent,
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
    "MATD3": MATD3_Agents,
}

__all__ = [
    "Agent", "MARLAgents", "RandomAgents",

    "OnPolicyAgent", "OffPolicyAgent", "OffPolicyMARLAgents", "OnPolicyMARLAgents",

    "REGISTRY_Agents",

    "PG_Agent", "A2C_Agent", "PPOCLIP_Agent", "PPOKL_Agent", "PPG_Agent", "DDPG_Agent", "TD3_Agent", "PDQN_Agent",
    "MPDQN_Agent", "SPDQN_Agent", "SAC_Agent", "SACDIS_Agent", "DQN_Agent", "DuelDQN_Agent", "DDQN_Agent",
    "NoisyDQN_Agent", "C51_Agent", "QRDQN_Agent", "PerDQN_Agent", "DRQN_Agent",

    "IQL_Agents", "VDN_Agents", "QMIX_Agents", "WQMIX_Agents", "QTRAN_Agents", "DCG_Agents", "VDAC_Agents",
    "COMA_Agents", "IDDPG_Agents", "MADDPG_Agents", "MFQ_Agents", "MFAC_Agents", "IPPO_Agents", "MAPPO_Agents",
    "ISAC_Agents", "MASAC_Agents", "MATD3_Agents"
]
