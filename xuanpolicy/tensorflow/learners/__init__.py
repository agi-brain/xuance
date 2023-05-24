import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import copy
import time
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Sequence, Optional, Callable, Union, Dict
from abc import ABC, abstractmethod
from argparse import Namespace

from .learner import Learner, LearnerMAS
from .policy_gradient.pg_learner import PG_Learner                      # 1.2
from .policy_gradient.a2c_learner import A2C_Learner                    # 1.3
from .policy_gradient.ppoclip_learner import PPOCLIP_Learner            # 1.4
from .policy_gradient.ppokl_learner import PPOKL_Learner
from .policy_gradient.ppg_learner import PPG_Learner
from .policy_gradient.ddpg_learner import DDPG_Learner                  # 1.5
from .policy_gradient.td3_learner import TD3_Learner                    # 1.6
from .policy_gradient.sac_learner import SAC_Learner                    # 1.10
from .policy_gradient.sacdis_learner import SACDIS_Learner
from .policy_gradient.pdqn_learner import PDQN_Learner
from .policy_gradient.mpdqn_learner import MPDQN_Learner
from .policy_gradient.spdqn_learner import SPDQN_Learner

from .qlearning_family.dqn_learner import DQN_Learner                   # 1.7
from .qlearning_family.dueldqn_learner import DuelDQN_Learner           # 1.8
from .qlearning_family.ddqn_learner import DDQN_Learner                 # 1.9
from .qlearning_family.perdqn_learner import PerDQN_Learner
from .qlearning_family.c51_learner import C51_Learner
from .qlearning_family.qrdqn_learner import QRDQN_Learner
from .qlearning_family.cdqn_learner import CDQN_Learner
from .qlearning_family.ldqn_learner import LDQN_Learner
from .qlearning_family.cldqn_learner import CLDQN_Learner

from .multi_agent_rl.iql_learner import IQL_Learner                     # 2.1
from .multi_agent_rl.vdn_learner import VDN_Learner                     # 2.2
from .multi_agent_rl.qmix_learner import QMIX_Learner                   # 2.3
from .multi_agent_rl.wqmix_learner import WQMIX_Learner                 # 2.4
from .multi_agent_rl.qtran_learner import QTRAN_Learner                 # 2.5
from .multi_agent_rl.dcg_learner import DCG_Learner                     # 2.6
from .multi_agent_rl.vdac_learner import VDAC_Learner                   # 2.7
from .multi_agent_rl.coma_learner import COMA_Learner                   # 2.8
from .multi_agent_rl.iddpg_learner import IDDPG_Learner                 # 2.9
from .multi_agent_rl.maddpg_learner import MADDPG_Learner               # 2.10
from .multi_agent_rl.mfq_learner import MFQ_Learner                     # 2.11
from .multi_agent_rl.mfac_learner import MFAC_Learner                   # 2.12
from .multi_agent_rl.mappoclip_learner import MAPPO_Clip_Learner        # 2.13
from .multi_agent_rl.mappokl_learner import MAPPO_KL_Learner            # 2.14
from .multi_agent_rl.isac_learner import ISAC_Learner                   # 2.15
from .multi_agent_rl.masac_learner import MASAC_Learner                 # 2.16
from .multi_agent_rl.madqn_learner import MADQN_Learner                 # 2.17
from .multi_agent_rl.maac_learner import MAAC_Learner                   # 2.18
from .multi_agent_rl.matd3_learner import MATD3_Learner


