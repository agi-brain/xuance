from typing import Optional, Union, List, Dict, Sequence, Callable, Any, Tuple, SupportsFloat, Type, Mapping
from xuance.common.common_tools import EPS, recursive_dict_update, get_configs, get_arguments, get_runner, \
    create_directory, combined_shape, space2shape, discount_cumsum, get_time_string
from xuance.common.statistic_tools import mpi_mean, mpi_moments, RunningMeanStd
from xuance.common.memory_tools import create_memory, store_element, sample_batch, Buffer, EpisodeBuffer, \
    DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari, DummyOffPolicyBuffer, DummyOffPolicyBuffer_Atari, \
    RecurrentOffPolicyBuffer, PerOffPolicyBuffer, SequentialReplayBuffer
from xuance.common.memory_tools_marl import BaseBuffer, MARL_OnPolicyBuffer, MARL_OnPolicyBuffer_RNN, \
    MeanField_OnPolicyBuffer, MeanField_OnPolicyBuffer_RNN, \
    MeanField_OffPolicyBuffer, MeanField_OffPolicyBuffer_RNN, \
    MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN
from xuance.common.memory_offline import OfflineBuffer_D4RL
from xuance.common.segtree_tool import SegmentTree, SumSegmentTree, MinSegmentTree

__all__ = [
    # typing
    "Optional", "Union", "List", "Dict", "Sequence", "Callable", "Any", "Tuple", "SupportsFloat", "Type", "Mapping",
    # common_tools
    "EPS", "recursive_dict_update", "get_configs", "get_arguments", "get_runner", "create_directory", "combined_shape",
    "space2shape", "discount_cumsum", "get_time_string",
    # statistic_tools
    "mpi_mean", "mpi_moments", "RunningMeanStd",
    # memory_tools
    "create_memory", "store_element", "sample_batch", "Buffer", "EpisodeBuffer",
    "DummyOnPolicyBuffer", "DummyOnPolicyBuffer_Atari", "DummyOffPolicyBuffer", "DummyOffPolicyBuffer_Atari",
    "RecurrentOffPolicyBuffer", "PerOffPolicyBuffer",
    "SequentialReplayBuffer",
    # memory_tools_marl
    "BaseBuffer", "MARL_OnPolicyBuffer", "MARL_OnPolicyBuffer_RNN", "MARL_OffPolicyBuffer", "MARL_OffPolicyBuffer_RNN",
    "MeanField_OnPolicyBuffer", "MeanField_OnPolicyBuffer_RNN",
    "MeanField_OffPolicyBuffer", "MeanField_OffPolicyBuffer_RNN",
    "I3CNet_Buffer", "I3CNet_Buffer_RNN",
    "OfflineBuffer_D4RL",
    # segtree_tool
    "SegmentTree", "SumSegmentTree", "MinSegmentTree",
]

try:
    from xuance.common.tuning_tools import set_hyperparameters, HyperParameterTuner, MultiObjectiveTuner

    __all__.append("set_hyperparameters")
    __all__.append("HyperParameterTuner")
    __all__.append("MultiObjectiveTuner")
except:
    pass

try:
    from xuance.common.offline_util import load_d4rl_dataset, compute_mean_std, normalize_states, return_range

    __all__.append("load_d4rl_dataset")
    __all__.append("compute_mean_std")
    __all__.append("normalize_states")
    __all__.append("return_range")
except:
    pass
