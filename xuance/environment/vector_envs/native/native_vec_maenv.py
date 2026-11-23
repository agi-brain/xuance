import numpy as np
from abc import abstractmethod
from xuance.environment.utils import space2shape
from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError


class NativeVecMultiAgentEnv(VecEnv):
    """Base class for natively batched/vectorized multi-agent simulators.

    Subclasses wrap a single simulator instance that executes ``num_envs``
    environments in parallel (e.g., PyTorch/GPU-batched simulators such as VMAS)
    while exposing XuanCe's standard vectorized MARL environment contract.
    """

    def __init__(self, num_envs, observation_space, action_space):
        super().__init__(num_envs, observation_space, action_space)
        self.waiting = False
        self.actions = None

        self.agents = getattr(self, "agents", None)
        self.num_agents = getattr(self, "num_agents", None)
        self.state_space = getattr(self, "state_space", None)
        self.env_info = getattr(self, "env_info", None)
        self.groups_info = getattr(self, "groups_info", None)
        self.max_episode_steps = getattr(self, "max_episode_steps", None)

        self.buf_state = []
        self.buf_obs = [{} for _ in range(self.num_envs)]
        self.buf_avail_actions = [{} for _ in range(self.num_envs)]
        self.buf_info = [{} for _ in range(self.num_envs)]

    def _init_buffers(self):
        """Allocate per-environment buffer lists after spaces are populated."""
        if self.state_space is not None:
            state_shape = space2shape(self.state_space)
            self.buf_state = [np.zeros(state_shape, dtype=np.float32) for _ in range(self.num_envs)]
        else:
            self.buf_state = [None for _ in range(self.num_envs)]
        self.buf_obs = [{} for _ in range(self.num_envs)]
        self.buf_avail_actions = [{} for _ in range(self.num_envs)]
        self.buf_info = [{} for _ in range(self.num_envs)]

    def step_async(self, actions):
        """Queue batched multi-agent actions for the next vectorized step."""
        if self.waiting:
            raise AlreadySteppingError
        listify = True
        try:
            if not isinstance(actions, dict) and len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if not listify:
            self.actions = list(actions)
        else:
            assert self.num_envs == 1, (
                f"actions {actions} is either not a list or has a wrong size - "
                f"cannot match to {self.num_envs} environments"
            )
            self.actions = [actions]
        self.waiting = True

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step_wait(self):
        raise NotImplementedError

    @abstractmethod
    def close_extras(self):
        raise NotImplementedError
