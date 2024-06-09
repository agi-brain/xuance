from .core import BasicQhead
from .core import ActorNet
from .core import CategoricalActorNet
from .core import CategoricalActorNet_SAC
from .core import GaussianActorNet
from .core import CriticNet
from .core import GaussianActorNet_SAC
from .core import VDN_mixer
from .core import QMIX_mixer
from .core import QMIX_FF_mixer
from .core import QTRAN_base
from .core import QTRAN_alt

from .categorical import ActorCriticPolicy as Categorical_AC_Policy
from .categorical import ActorPolicy as Categorical_Actor_Policy
from .categorical import PPGActorCritic as Categorical_PPG_Policy
from .categorical import SACDISPolicy as Categorical_SAC_Policy

from .gaussian import ActorCriticPolicy as Gaussian_AC_Policy
from .gaussian import ActorPolicy as Gaussian_Actor_Policy
from .gaussian import PPGActorCritic as Gaussian_PPG_Policy
from .gaussian import SACPolicy as Gaussian_SAC_Policy

from .deterministic import BasicQnetwork
from .deterministic import DuelQnetwork
from .deterministic import NoisyQnetwork
from .deterministic import C51Qnetwork
from .deterministic import QRDQN_Network
from .deterministic import DDPGPolicy
from .deterministic import TD3Policy
from .deterministic import PDQNPolicy
from .deterministic import MPDQNPolicy
from .deterministic import SPDQNPolicy
from .deterministic import DRQNPolicy

from .deterministic_marl import BasicQnetwork as BasicQnetwork_marl
from .deterministic_marl import MFQnetwork, MixingQnetwork, Weighted_MixingQnetwork, Qtran_MixingQnetwork, DCG_policy, \
    Independent_DDPG_Policy, MADDPG_Policy, MATD3_Policy
from .categorical_marl import MeanFieldActorCriticPolicy, COMA_Policy
from .categorical_marl import MAAC_Policy as Categorical_MAAC_Policy
from .categorical_marl import MAAC_Policy_Share as Categorical_MAAC_Policy_Share
from .gaussian_marl import Basic_ISAC_Policy as Gaussian_ISAC
from .gaussian_marl import MASAC_Policy as Gaussian_MASAC
from .gaussian_marl import MAAC_Policy as Gaussain_MAAC

Mixer = {
    "VDN": VDN_mixer,
    "QMIX": QMIX_mixer,
    "WQMIX": QMIX_FF_mixer,
    "QTRAN_alt": QTRAN_alt,
    "QTRAN_base": QTRAN_base
}

REGISTRY_Policy = {
    # ↓ Single-Agent DRL ↓ #
    "Categorical_AC": Categorical_AC_Policy,
    "Categorical_Actor": Categorical_Actor_Policy,
    "Categorical_PPG": Categorical_PPG_Policy,
    "Gaussian_AC": Gaussian_AC_Policy,
    "Gaussian_SAC": Gaussian_SAC_Policy,
    "Discrete_SAC": Categorical_SAC_Policy,
    "Gaussian_PPG": Gaussian_PPG_Policy,
    "Gaussian_Actor": Gaussian_Actor_Policy,
    "Basic_Q_network": BasicQnetwork,
    "Duel_Q_network": DuelQnetwork,
    "Noisy_Q_network": NoisyQnetwork,
    "C51_Q_network": C51Qnetwork,
    "QR_Q_network": QRDQN_Network,
    "DDPG_Policy": DDPGPolicy,
    "TD3_Policy": TD3Policy,
    "PDQN_Policy": PDQNPolicy,
    "MPDQN_Policy": MPDQNPolicy,
    "SPDQN_Policy": SPDQNPolicy,
    "DRQN_Policy": DRQNPolicy,
    # ↓ Multi-Agent DRL ↓ #
    "Basic_Q_network_marl": BasicQnetwork_marl,
    "Mixing_Q_network": MixingQnetwork,
    "Weighted_Mixing_Q_network": Weighted_MixingQnetwork,
    "Qtran_Mixing_Q_network": Qtran_MixingQnetwork,
    "DCG_Policy": DCG_policy,
    "Categorical_MAAC_Policy": Categorical_MAAC_Policy,
    "Categorical_MAAC_Policy_Share": Categorical_MAAC_Policy_Share,
    "Categorical_COMA_Policy": COMA_Policy,
    "Independent_DDPG_Policy": Independent_DDPG_Policy,
    "MADDPG_Policy": MADDPG_Policy,
    "MF_Q_network": MFQnetwork,
    "Categorical_MFAC_Policy": MeanFieldActorCriticPolicy,
    "Gaussian_MAAC_Policy": Gaussain_MAAC,
    "Gaussian_ISAC_Policy": Gaussian_ISAC,
    "Gaussian_MASAC_Policy": Gaussian_MASAC,
    "MATD3_Policy": MATD3_Policy
}

__all__ = [
    "REGISTRY_Policy", "Mixer",
    "ActorNet", "CategoricalActorNet", "CategoricalActorNet_SAC", "GaussianActorNet", "GaussianActorNet_SAC",
    "BasicQhead", "CriticNet", "GaussianActorNet_SAC",
    "VDN_mixer", "QMIX_mixer", "QMIX_FF_mixer", "QTRAN_base", "QTRAN_alt",
    "Categorical_AC_Policy", "Categorical_Actor_Policy", "Categorical_PPG_Policy", "Categorical_SAC_Policy",
    "Gaussian_AC_Policy", "Gaussian_Actor_Policy", "Gaussian_PPG_Policy", "Gaussian_SAC_Policy",
    "BasicQnetwork", "DuelQnetwork", "NoisyQnetwork", "C51Qnetwork", "QRDQN_Network", "DDPGPolicy", "TD3Policy",
    "PDQNPolicy", "MPDQNPolicy", "SPDQNPolicy", "DRQNPolicy",
    "BasicQnetwork_marl", "MFQnetwork", "MixingQnetwork", "Weighted_MixingQnetwork", "Qtran_MixingQnetwork",
    "DCG_policy", "Independent_DDPG_Policy", "MADDPG_Policy", "MATD3_Policy",
    "MeanFieldActorCriticPolicy", "COMA_Policy", "Categorical_MAAC_Policy", "Categorical_MAAC_Policy_Share",
    "Gaussian_ISAC", "Gaussian_MASAC", "Gaussain_MAAC",
]
