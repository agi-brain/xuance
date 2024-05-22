from abc import ABC, abstractmethod


class MLPPolicy(ABC):
    """Abstract MLP policy class. Computes actions given relevant information."""

    @abstractmethod
    def get_actions(self, obs, available_actions, t_env, explore):
        """
        Compute actions using the needed information.
        :param obs: (np.ndarray) Observations with which to compute actions.
        :param rnn_states: (np.ndarray / torch.Tensor) RNN state to use to compute actions
        :param available_actions: (np.ndarray) contains actions which are available to take. If None, there are no action restrictions.
        :param t_env: (int) train step during which this function is called. Used to compute epsilon for eps-greedy exploration.
        :param explore: (bool) whether to return actions using an exploration policy.

        :return: (torch.Tensor / np.ndarray) computed actions (np.ndarray if explore is True, torch.Tensor else)
        :return: (torch.Tensor) additional information, depending on algorithms (e.g. action entropy for RMASAC).
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_actions(self, obs, available_actions):
        """
        Compute actions uniformly at random.
        :param obs: (np.ndarray) Current observation corresponding to actions.
        :param prev_actions: (np.ndarray) Optionally use previous action to  compute actions.

        :return: (np.ndarray) random actions
        """
        raise NotImplementedError
