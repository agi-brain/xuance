from tqdm import tqdm
import numpy as np
from copy import deepcopy
from typing import Optional
from argparse import Namespace
from xuance.torch.agents.base import Agent
from xuance.environment import DummyVecEnv
from xuance.common import DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari
from xuance.torch.utils import split_distributions


class OffPolicyAgent(Agent):
    """The core class for on-policy algorithm with single agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(OffPolicyAgent, self).__init__(config, envs)

    def _build_memory(self, auxiliary_info_shape=None):
        return

    def _build_policy(self):
        raise NotImplementedError

    def get_terminated_values(self, observations_next: np.ndarray, rewards: np.ndarray = None):
        """Returns values for terminated states.

        Parameters:
            observations_next (np.ndarray): The terminal observations.
            rewards (np.ndarray): The rewards for terminated states.

        Returns:
            values_next: The values for terminal states.
        """
        return

    def action(self, observations: np.ndarray,
               e_greedy: float = None, noise_scale: float = None):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            e_greedy (float): The epsilon greedy.
            noise_scale (float): The scale of noise.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        return

    def train_epochs(self, n_epochs=1):
        return

    def train(self, train_steps):
        return

    def test(self, env_fn, test_episodes):
        return

