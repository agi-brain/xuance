from abc import ABC, abstractmethod
from gym.spaces import Space, Box, Discrete, Dict
from argparse import Namespace
from mpi4py import MPI
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

import tensorflow as tf
import tensorflow.keras as tk

from xuance.environment import *
from xuance.common import *
from xuance.tensorflow.learners import *
from xuance.tensorflow.policies import *
from xuance.tensorflow.utils import *
from xuance.tensorflow.policies import REGISTRY as REGISTRY_Policy
from xuance.tensorflow.utils.input_reformat import get_repre_in, get_policy_in_marl
from xuance.tensorflow.representations import REGISTRY as REGISTRY_Representation
from xuance.tensorflow.runners.runner_basic import MyLinearLR

from .agent import Agent, get_total_iters
from .agents_marl import MARLAgents, RandomAgents
from .agent import get_total_iters

'''
Single-Agent DRL Algorithms
'''
from .policy_gradient.pg_agent import PG_Agent
from .policy_gradient.a2c_agent import A2C_Agent
from .policy_gradient.ppoclip_agent import PPOCLIP_Agent
from .policy_gradient.ppokl_agent import PPOKL_Agent
from .policy_gradient.ppg_agent import PPG_Agent
from .policy_gradient.ddpg_agent import DDPG_Agent
from .policy_gradient.td3_agent import TD3_Agent
from .policy_gradient.pdqn_agent import PDQN_Agent
from .policy_gradient.mpdqn_agent import MPDQN_Agent
from .policy_gradient.spdqn_agent import SPDQN_Agent
from .policy_gradient.sac_agent import SAC_Agent
from .policy_gradient.sacdis_agent import SACDIS_Agent

from .qlearning_family.dqn_agent import DQN_Agent
from .qlearning_family.dueldqn_agent import DuelDQN_Agent
from .qlearning_family.ddqn_agent import DDQN_Agent
from .qlearning_family.noisydqn_agent import NoisyDQN_Agent
from .qlearning_family.c51_agent import C51_Agent
from .qlearning_family.qrdqn_agent import QRDQN_Agent
from .qlearning_family.perdqn_agent import PerDQN_Agent
from .qlearning_family.drqn_agent import DRQN_Agent

'''
Multi-Agent DRL algorithms
'''
from .multi_agent_rl.iql_agents import IQL_Agents
from .multi_agent_rl.vdn_agents import VDN_Agents
from .multi_agent_rl.qmix_agents import QMIX_Agents
from .multi_agent_rl.wqmix_agents import WQMIX_Agents
from .multi_agent_rl.qtran_agents import QTRAN_Agents
from .multi_agent_rl.dcg_agents import DCG_Agents
from .multi_agent_rl.vdac_agents import VDAC_Agents
from .multi_agent_rl.coma_agents import COMA_Agents
from .multi_agent_rl.iddpg_agents import IDDPG_Agents
from .multi_agent_rl.maddpg_agents import MADDPG_Agents
from .multi_agent_rl.mfq_agents import MFQ_Agents
from .multi_agent_rl.mfac_agents import MFAC_Agents
from .multi_agent_rl.ippo_agents import IPPO_Agents
from .multi_agent_rl.mappo_agents import MAPPO_Agents
from .multi_agent_rl.isac_agents import ISAC_Agents
from .multi_agent_rl.masac_agents import MASAC_Agents
from .multi_agent_rl.matd3_agents import MATD3_Agents

REGISTRY = {
    "PG": PG_Agent,
    "A2C": A2C_Agent,
    "PPO_Clip": PPOCLIP_Agent,
    "PPO_KL": PPOKL_Agent,
    "PPG": PPG_Agent,
    "PDQN": PDQN_Agent,
    "MPDQN": MPDQN_Agent,
    "SPDQN": SPDQN_Agent,
    "DDPG": DDPG_Agent,
    "SAC": SAC_Agent,
    "SACDIS":SACDIS_Agent,
    "TD3": TD3_Agent,
    "DQN": DQN_Agent,
    "Duel_DQN": DuelDQN_Agent,
    "DDQN": DDQN_Agent,
    "NoisyDQN": NoisyDQN_Agent,
    "PerDQN": PerDQN_Agent,
    "C51DQN": C51_Agent,
    "QRDQN": QRDQN_Agent,
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
    # "MADQN": MADQN_Agents,
    # "MAAC": MAAC_Agents,
    "MATD3": MATD3_Agents,
}
