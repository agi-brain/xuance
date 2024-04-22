import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Optional, Callable, Union, Dict
from gym.spaces import Space, Box, Discrete, Dict
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace

from .learner import Learner, LearnerMAS
from .policy_gradient import (
    PG_Learner,
    A2C_Learner,
    PPOCLIP_Learner,
    PPOKL_Learner,
    PPG_Learner,
    DDPG_Learner,
    TD3_Learner,
    SAC_Learner,
    SACDIS_Learner,
    PDQN_Learner,
    MPDQN_Learner,
    SPDQN_Learner
)

from .qlearning_family import (
    DQN_Learner,
    DuelDQN_Learner,
    DDQN_Learner,
    PerDQN_Learner,
    C51_Learner,
    QRDQN_Learner,
    DRQN_Learner
)

from .multi_agent_rl import (
    IQL_Learner,
    VDN_Learner,
    QMIX_Learner,
    WQMIX_Learner,
    QTRAN_Learner,
    VDAC_Learner,
    COMA_Learner,
    IDDPG_Learner,
    MADDPG_Learner,
    MFQ_Learner,
    MFAC_Learner,
    IPPO_Learner,
    MAPPO_Clip_Learner,
    ISAC_Learner,
    MASAC_Learner,
    MATD3_Learner
)
