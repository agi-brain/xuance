from abc import ABC, abstractmethod
from gym.spaces import Space, Box, Discrete, Dict
from argparse import Namespace
from mpi4py import MPI
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from xuance.environment import *
from xuance.common import *
from xuance.torch.learners import *
from xuance.torch.policies import *
from xuance.torch.utils import *
from xuance.torch.policies import REGISTRY as REGISTRY_Policy
from xuance.torch.utils.input_reformat import get_repre_in, get_policy_in_marl
from xuance.torch.representations import REGISTRY as REGISTRY_Representation

from .agent import Agent, get_total_iters
from .agents_marl import MARLAgents, RandomAgents

'''
Single-Agent DRL algorithms
'''
from .policy_gradient import (
    PG_Agent,
    A2C_Agent,
    PPOCLIP_Agent,
    PPOKL_Agent,
    PPG_Agent,
    DDPG_Agent,
    TD3_Agent,
    PDQN_Agent,
    MPDQN_Agent,
    SPDQN_Agent,
    SAC_Agent,
    SACDIS_Agent
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
'''
Multi-Agent DRL algorithms
'''
from .multi_agent_rl import (
    IQL_Agents,
    VDN_Agents,
    QMIX_Agents,
    WQMIX_Agents,
    QTRAN_Agents,
    DCG_Agents,
    VDAC_Agents,
    COMA_Agents,
    IDDPG_Agents,
    MADDPG_Agents,
    MFQ_Agents,
    MFAC_Agents,
    IPPO_Agents,
    MAPPO_Agents,
    ISAC_Agents,
    MASAC_Agents,
    MATD3_Agents
)

REGISTRY = {
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
