from .env_utils import (
    tile_images,
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
    obs_n_space_info,
    clear_mpi_env_vars,
    flatten_list,
    flatten_obs,
    CloudpickleWrapper,
)

from .subprocess.subproc_vec_env import (
    worker,
    SubprocVecEnv
)

from .vector_env import (
    AlreadySteppingError,
    NotSteppingError,
    VecEnv
)

from .subprocess import SubprocVecEnv
from .subprocess import SubprocVecEnv_Atari
from .subprocess import SubprocVecMultiAgentEnv
from .subprocess import SubprocVecEnv_StarCraft2
from .subprocess import SubprocVecEnv_Football
from .dummy import DummyVecEnv
from .dummy import DummyVecEnv_Atari
from .dummy import DummyVecMultiAgentEnv
from .dummy import DummyVecEnv_StarCraft2
from .dummy import DummyVecEnv_Football

REGISTRY_VEC_ENV = {
    "DummyVecEnv": DummyVecEnv,
    "DummyVecMultiAgentEnv": DummyVecMultiAgentEnv,
    "Dummy_Atari": DummyVecEnv_Atari,
    "Dummy_StarCraft2": DummyVecEnv_StarCraft2,
    "Dummy_Football": DummyVecEnv_Football,

    # multiprocess #
    "SubprocVecEnv": SubprocVecEnv,
    "SubprocVecMultiAgentEnv": SubprocVecMultiAgentEnv,
    "Subproc_Atari": SubprocVecEnv_Atari,
    "Subproc_StarCraft2": SubprocVecEnv_StarCraft2,
    "Subproc_Football": SubprocVecEnv_Football,
}
