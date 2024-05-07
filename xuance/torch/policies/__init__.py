import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Sequence, Optional, Callable, Union
from gymnasium.spaces import Space, Box, Discrete, Dict

from .categorical import ActorCriticPolicy as Categorical_AC_Policy
from .categorical import ActorPolicy as Categorical_Actor_Policy
from .categorical import PPGActorCritic as Categorical_PPG_Policy
from .categorical import SACDISPolicy as Categorical_SAC_Policy

from .gaussian import ActorCriticPolicy as Gaussian_AC_Policy
from .gaussian import ActorPolicy as Gaussian_Actor_Policy
from .gaussian import PPGActorCritic as Gaussian_PPG_Policy
from .deterministic import BasicQnetwork, C51Qnetwork, DuelQnetwork, DDPGPolicy, NoisyQnetwork, QRDQN_Network, \
    TD3Policy, PDQNPolicy, MPDQNPolicy, SPDQNPolicy, DRQNPolicy
from .gaussian import SACPolicy as Gaussian_SAC_Policy

from .mixers import *
from .deterministic_marl import BasicQnetwork as BasicQnetwork_marl
from .deterministic_marl import Basic_DDPG_policy as BasicDDPG_marl
from .deterministic_marl import MFQnetwork, MixingQnetwork, Weighted_MixingQnetwork, Qtran_MixingQnetwork, DCG_policy, \
    Basic_DDPG_policy, MADDPG_policy, MATD3_policy
from .categorical_marl import MeanFieldActorCriticPolicy, COMAPolicy
from .categorical_marl import MAAC_Policy as Categorical_MAAC_Policy
from .categorical_marl import MAAC_Policy_Share as Categorical_MAAC_Policy_Share
from .gaussian_marl import Basic_ISAC_policy as Gaussian_ISAC
from .gaussian_marl import MASAC_policy as Gaussian_MASAC
from .gaussian_marl import MAAC_Policy as Gaussain_MAAC

Mixer = {
    "VDN": VDN_mixer,
    "QMIX": QMIX_mixer,
    "WQMIX": QMIX_FF_mixer,
    "QTRAN_alt": QTRAN_alt,
    "QTRAN_base": QTRAN_base
}

REGISTRY = {
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
    "Categorical_COMA_Policy": COMAPolicy,
    "Independent_DDPG_Policy": BasicDDPG_marl,
    "MADDPG_Policy": MADDPG_policy,
    "MF_Q_network": MFQnetwork,
    "Categorical_MFAC_Policy": MeanFieldActorCriticPolicy,
    "Gaussian_MAAC_Policy": Gaussain_MAAC,
    "Gaussian_ISAC_Policy": Gaussian_ISAC,
    "Gaussian_MASAC_Policy": Gaussian_MASAC,
    "MATD3_Policy": MATD3_policy
}

Policy_Inputs = {
    # DRL (discrete action spaces) #
    "Categorical_AC": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                       "normalize", "initialize", "activation", "device"],
    "Categorical_Actor": ["action_space", "representation", "actor_hidden_size",
                          "normalize", "initialize", "activation", "device"],
    "Discrete_SAC": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                     "normalize", "initialize", "activation", "device"],
    "Categorical_PPG": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                        "normalize", "initialize", "activation", "device"],
    "Basic_Q_network": ["action_space", "representation", "hidden_sizes",
                        "normalize", "initialize", "activation", "device"],
    "Duel_Q_network": ["action_space", "representation", "hidden_sizes",
                       "normalize", "initialize", "activation", "device"],
    "Noisy_Q_network": ["action_space", "representation", "hidden_sizes",
                        "normalize", "initialize", "activation", "device"],
    "C51_Q_network": ["action_space", "atom_num", "vmin", "vmax", "representation", "hidden_sizes",
                      "normalize", "initialize", "activation", "device"],
    "QR_Q_network": ["action_space", "quantile_num", "representation", "hidden_sizes",
                     "normalize", "initialize", "activation", "device"],
    # DRL (continuous action spaces) #
    "Gaussian_AC": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                    "normalize", "initialize", "activation", "activation_action", "device"],
    "Gaussian_SAC": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                     "normalize", "initialize", "activation", "activation_action", "device"],
    "Gaussian_Actor": ["action_space", "representation", "actor_hidden_size",
                       "normalize", "initialize", "activation", "activation_action", "device", "fixed_std"],
    "Gaussian_PPG": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                     "normalize", "initialize", "activation", "activation_action", "device"],
    "DDPG_Policy": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                    "initialize", "activation", "activation_action", "device"],
    "SAC_Policy": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                   "normalize", "initialize", "activation", "activation_action", "device"],
    "TD3_Policy": ["action_space", "representation", "actor_hidden_size", "critic_hidden_size",
                   "normalize", "initialize", "activation", "activation_action", "device"],
    # DRL (hybrid action spaces) #
    "PDQN_Policy": ['observation_space', 'action_space', 'representation', 'conactor_hidden_size',
                    'qnetwork_hidden_size', 'normalize', 'initialize', 'activation', 'activation_action', 'device'],
    "MPDQN_Policy": ['observation_space', 'action_space', 'representation', 'conactor_hidden_size',
                     'qnetwork_hidden_size', 'normalize', 'initialize', 'activation', 'activation_action', 'device'],
    "SPDQN_Policy": ['observation_space', 'action_space', 'representation', 'conactor_hidden_size',
                     'qnetwork_hidden_size', 'normalize', 'initialize', 'activation', 'activation_action', 'device'],
    #  MARL policies (discrete action spaces) #
    "Basic_Q_network_marl": ["action_space", "n_agents", "representation", "hidden_sizes",
                             "normalize", "initialize", "activation", "device"],
    "Mixing_Q_network": ["action_space", "n_agents", "representation", "mixer", "hidden_sizes",
                         "normalize", "initialize", "activation", "device"],
    "Weighted_Mixing_Q_network": ["action_space", "n_agents", "representation", "mixer", "ff_mixer", "hidden_sizes",
                                  "normalize", "initialize", "activation", "device"],
    "Qtran_Mixing_Q_network": ["action_space", "n_agents", "representation", "mixer", "qtran_mixer", "hidden_sizes",
                               "normalize", "initialize", "activation", "device"],
    "Categorical_MAAC_Policy": ["action_space", "n_agents", "representation", "mixer", "actor_hidden_size",
                                "critic_hidden_size", "normalize", "initialize", "activation", "device"],
    "Categorical_MAAC_Policy_Share": ["action_space", "n_agents", "representation", "mixer", "actor_hidden_size",
                                      "critic_hidden_size", "normalize", "initialize", "activation", "device"],
    "Categorical_MFAC_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size",
                                "critic_hidden_size", "normalize", "initialize", "activation", "device"],
    "Categorical_COMA_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size",
                                "critic_hidden_size", "normalize", "initialize", "activation", "device"],
    "MF_Q_network": ["action_space", "n_agents", "representation", "hidden_sizes",
                     "normalize", "initialize", "activation", "device"],
    #  MARL policies (continuous action spaces) #
    "Independent_DDPG_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size",
                                "critic_hidden_size", "normalize", "initialize", "activation", "activation_action",
                                "device"],
    "MADDPG_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size", "critic_hidden_size",
                      "normalize", "initialize", "activation", "activation_action", "device"],
    "Gaussian_MAAC_Policy": ["action_space", "n_agents", "representation", "mixer", "actor_hidden_size",
                             "critic_hidden_size", "normalize", "initialize", "activation", "activation_action",
                             "device"],
    "Gaussian_ISAC_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size",
                             "critic_hidden_size", "normalize", "initialize", "activation", "activation_action",
                             "device"],
    "Gaussian_MASAC_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size", "critic_hidden_size",
                              "normalize", "initialize", "activation", "activation_action", "device"],
    "MATD3_Policy": ["action_space", "n_agents", "representation", "actor_hidden_size", "critic_hidden_size",
                     "normalize", "initialize", "activation", "activation_action", "device"],
}

Policy_Inputs_All = {
    "state_dim": None,
    "action_space": None,
    "n_agents": 1,
    "representation": None,
    "mixer": None,
    "ff_mixer": None,
    "qtran_mixer": None,
    "hidden_sizes": None,
    "actor_hidden_size": None,
    "critic_hidden_size": None,
    "normalize": None,
    "initialize": None,
    "activation": None,
    "activation_action": None,
    "device": None,
    "fixed_std": None
}
