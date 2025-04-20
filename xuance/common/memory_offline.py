import numpy as np
from gymnasium import Space
from xuance.common import Optional, space2shape
from xuance.common.memory_tools import Buffer, create_memory, store_element, sample_batch


class OfflineBuffer_D4RL(Buffer):
    """
    Replay buffer for OfflineBuffer DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        buffer_size: the total size of the replay buffer.
        batch_size: size of transition data for a batch of sample.
    """

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 buffer_size: int,
                 batch_size: int,
                 ):
        super(OfflineBuffer_D4RL, self).__init__(observation_space, action_space, auxiliary_shape)
        self.n_envs, self.batch_size = n_envs, batch_size
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = buffer_size // self.n_envs
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)

    def d4rl2buffer(self, dataset):
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']

        next_observations = dataset['next_observations']
        terminals = dataset['terminals']

        num_samples = observations.shape[0]
        assert num_samples <= self.n_size, "Dataset size exceeds buffer capacity."
        for env_idx in range(self.n_envs):

            if isinstance(self.observations, dict):
                for key in self.observations.keys():
                    self.observations[key][env_idx, :num_samples] = observations
            else:
                self.observations[env_idx, :num_samples] = observations

            if isinstance(self.actions, dict):
                for key in self.actions.keys():
                    self.actions[key][env_idx, :num_samples] = actions
            else:
                self.actions[env_idx, :num_samples] = actions

            self.rewards[env_idx, :num_samples] = rewards

            if isinstance(self.next_observations, dict):
                for key in self.next_observations.keys():
                    self.next_observations[key][env_idx, :num_samples] = next_observations
            else:
                self.next_observations[env_idx, :num_samples] = next_observations

            self.terminals[env_idx, :num_samples] = terminals

        self.ptr = num_samples
        self.size = num_samples

    def store(self, obs, acts, rews, terminals, next_obs):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(next_obs, self.next_observations, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)

    def sample(self, batch_size=None):
        bs = self.batch_size if batch_size is None else batch_size
        env_choices = np.random.choice(self.n_envs, bs)
        step_choices = np.random.choice(self.size, bs)

        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'obs_next': sample_batch(self.next_observations, tuple([env_choices, step_choices])),
            'rewards': sample_batch(self.rewards, tuple([env_choices, step_choices])),
            'terminals': sample_batch(self.terminals, tuple([env_choices, step_choices])),
            'batch_size': bs,
        }
        return samples_dict
