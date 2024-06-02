from xuance.environment.vector_envs.env_utils import (
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

from xuance.environment.vector_envs.subprocess.subproc_vec_env import (
    worker,
    SubprocVecEnv
)

from xuance.environment.vector_envs.vector_env import (
    AlreadySteppingError,
    NotSteppingError,
    VecEnv
)

from xuance.environment.vector_envs.subprocess import SubprocVecEnv
from xuance.environment.vector_envs.subprocess import SubprocVecEnv_Atari
from xuance.environment.vector_envs.subprocess import SubprocVecMultiAgentEnv
from xuance.environment.vector_envs.dummy import DummyVecEnv
from xuance.environment.vector_envs.dummy import DummyVecEnv_Atari
from xuance.environment.vector_envs.dummy import DummyVecMultiAgentEnv

REGISTRY_VEC_ENV = {
    "DummyVecEnv": DummyVecEnv,
    "DummyVecMultiAgentEnv": DummyVecMultiAgentEnv,
    "Dummy_Atari": DummyVecEnv_Atari,

    # multiprocess #
    "SubprocVecEnv": SubprocVecEnv,
    "SubprocVecMultiAgentEnv": SubprocVecMultiAgentEnv,
    "Subproc_Atari": SubprocVecEnv_Atari,
}
