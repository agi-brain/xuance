from .learner import Learner, LearnerMAS
from .policy_gradient import PG_Learner, A2C_Learner, PPOCLIP_Learner, PPOKL_Learner, PPG_Learner, DDPG_Learner, \
    TD3_Learner, SAC_Learner, SACDIS_Learner, PDQN_Learner, MPDQN_Learner, SPDQN_Learner,NPG_Learner

from .qlearning_family import DQN_Learner, DuelDQN_Learner, DDQN_Learner, PerDQN_Learner, C51_Learner, QRDQN_Learner, \
    DRQN_Learner

from .model_based import DreamerV2_Learner, DreamerV3_Learner
from .contrastive_unsupervised_rl import CURL_Learner,SPR_Learner,DrQ_Learner

from .multi_agent_rl import (IQL_Learner, VDN_Learner, QMIX_Learner, WQMIX_Learner, QTRAN_Learner, \
    IAC_Learner, VDAC_Learner, COMA_Learner, MFQ_Learner, MFAC_Learner, IPPO_Learner, MAPPO_Clip_Learner, \
    IC3Net_Learner, CommNet_Learner, TarMAC_Learner, \
    IDDPG_Learner, MADDPG_Learner, MATD3_Learner, ISAC_Learner, ISACDIS_Learner, MASAC_Learner, MASACDIS_Learner)

from .offline import TD3_BC_Learner

REGISTRY_Learners = {
    "BasicLearner": Learner,
    "BasicLearnerMAS": LearnerMAS,
    "PG_Learner": PG_Learner,
    "A2C_Learner": A2C_Learner,
    "PPOCLIP_Learner": PPOCLIP_Learner,
    "PPOKL_Learner": PPOKL_Learner,
    "PPG_Learner": PPG_Learner,
    "DDPG_Learner": DDPG_Learner,
    "TD3_Learner": TD3_Learner,
    "SAC_Learner": SAC_Learner,
    "SACDIS_Learner": SACDIS_Learner,
    "PDQN_Learner": PDQN_Learner,
    "MPDQN_Learner": MPDQN_Learner,
    "SPDQN_Learner": SPDQN_Learner,
    "NPG_Learner": NPG_Learner,

    "DQN_Learner": DQN_Learner,
    "DuelDQN_Learner": DuelDQN_Learner,
    "DDQN_Learner": DDQN_Learner,
    "PerDQN_Learner": PerDQN_Learner,
    "C51_Learner": C51_Learner,
    "QRDQN_Learner": QRDQN_Learner,
    "DRQN_Learner": DRQN_Learner,

    "DreamerV2_Learner": DreamerV2_Learner,
    "DreamerV3_Learner": DreamerV3_Learner,

    "IQL_Learner": IQL_Learner,
    "VDN_Learner": VDN_Learner,
    "QMIX_Learner": QMIX_Learner,
    "WQMIX_Learner": WQMIX_Learner,
    "QTRAN_Learner": QTRAN_Learner,
    "IAC_Learner": IAC_Learner,
    "VDAC_Learner": VDAC_Learner,
    "COMA_Learner": COMA_Learner,
    "IDDPG_Learner": IDDPG_Learner,
    "MADDPG_Learner": MADDPG_Learner,
    "MFQ_Learner": MFQ_Learner,
    "MFAC_Learner": MFAC_Learner,
    "IPPO_Learner": IPPO_Learner,
    "MAPPO_Clip_Learner": MAPPO_Clip_Learner,
    "ISAC_Learner": ISAC_Learner,
    "ISACDIS_Learner": ISACDIS_Learner,
    "MASAC_Learner": MASAC_Learner,
    "MASACDIS_Learner": MASACDIS_Learner,
    "MATD3_Learner": MATD3_Learner,

    "CommNet_Learner": CommNet_Learner,
    "IC3Net_Learner": IC3Net_Learner,
    "TarMac_Learner": TarMAC_Learner,

    "TD3BC_Learner": TD3_BC_Learner,

    "CURL_Learner": CURL_Learner,
    "SPR_Learner": SPR_Learner,
    "DrQ_Learner": DrQ_Learner
}

__all__ = [
    "REGISTRY_Learners", "Learner", "LearnerMAS",

    "PG_Learner", "A2C_Learner", "PPOCLIP_Learner", "PPOKL_Learner", "PPG_Learner", "DDPG_Learner", "TD3_Learner",
    "SAC_Learner", "SACDIS_Learner", "PDQN_Learner", "MPDQN_Learner", "SPDQN_Learner","NPG_Learner",

    "DQN_Learner", "DuelDQN_Learner", "DDQN_Learner", "PerDQN_Learner", "C51_Learner", "QRDQN_Learner", "DRQN_Learner",

    "DreamerV2_Learner", "DreamerV3_Learner",

    "IQL_Learner", "VDN_Learner", "QMIX_Learner", "WQMIX_Learner", "QTRAN_Learner",
    "IAC_Learner", "VDAC_Learner", "COMA_Learner", "IPPO_Learner", "MAPPO_Clip_Learner",
    "MFQ_Learner", "MFAC_Learner",
    "IDDPG_Learner", "MADDPG_Learner", "MATD3_Learner",
    "ISAC_Learner", "ISACDIS_Learner", "MASAC_Learner", "MADDPG_Learner",
    "IC3Net_Learner", "CommNet_Learner", "TarMAC_Learner",

    "TD3_BC_Learner",

    "CURL_Learner","SPR_Learner","DrQ_Learner"
]
