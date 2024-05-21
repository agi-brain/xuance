from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from gym import spaces

AgentID = Any
AgentValue = Any
MultiAgentDict = Dict[AgentID, AgentValue]
AgentKeys = List[str]


class RawEnvironment(ABC):
    """
    A base class for new environment.

    The following attributes are necessary when creating a new environment:
        - self.env: the environment object;
        - self.observation_space: the observation space of the agent;
        - self.action_space: the action space of the agent;
        - self.max_episode_steps: the maximum steps for one episode of the environment in XuanCe.
    """

    def __new__(cls, *args, **kwargs):
        if cls is RawEnvironment:
            raise TypeError("Type RawEnvironment cannot be instantiated; It can be used only as a base class")
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        super(RawEnvironment, self).__init__(*args, **kwargs)
        self.env = None
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        self.max_episode_steps: Optional[int] = None

    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the environment with kwargs.

        Returns:
            observation (np.ndarray or list): The initial observations of the agent.
            info (dict): The information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Steps through the environment with action.

        Parameters:
            action (np.ndarray or list): The action to be executed.

        Return:
            observation (np.ndarray or list): The next step observation after executing action.
            reward (np.ndarray or list): The reward returned by the environment.
            terminated(np.ndarray or list): A bool value that indicates if the environment should be terminated.
            truncated(np.ndarray or list): A bool value that indicates if the environment should be truncated.
            info (dict): The information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, *args, **kwargs):
        """
        Renders the environment.

        Return:
            rgb_images (np.ndarray or list): The images used to visualize the environment.
        """
        return NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the environment."""
        return NotImplementedError


class RawMultiAgentEnv(ABC):
    """A base class for multi-agent environment.

    The following attributes are necessary when creating a new multi-agent environment in XuanCe:
        - self.env: the environment object;
        - self.observation_space: the observation space of the agent;
        - self.action_space: the action space of the agent;
        - self.agents: a list of all agents' ids;
        - self.num_agents: the number of total agents in the environment;
        - self.groups: a list of groups. Each group contains agents' ids with a same role;
        - self.num_groups: the number of groups of the environment, default is 1;
        - self.max_episode_steps: the maximum steps for one episode of the environment.
    """

    def __new__(cls, *args, **kwargs):
        if cls is RawMultiAgentEnv:
            raise TypeError("Type RawMultiAgentEnv cannot be instantiated; It can be used only as a base class")
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        super(RawMultiAgentEnv, self).__init__(*args, **kwargs)
        self.env = None
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        self.agents: Optional[AgentKeys] = ["agent_1"]
        self.num_agents: Optional[int] = 1
        self.group: Optional[List[AgentKeys]] = [["agent"]]
        self.num_groups: Optional[int] = 1
        self.max_episode_steps: Optional[int] = None

    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the environment with kwargs.

        Returns:
            observation (np.ndarray or list): The initial observations of the agent.
            info (dict): The information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Steps through the environment with action.

        Parameters:
            action_dict (MultiAgentDict): A dict that contains all agents' actions.

        Return:
            observation (MultiAgentDict): The next step observations after executing actions.
            reward (MultiAgentDict): The rewards returned by the environment.
            terminated(MultiAgentDict): A bool values that indicates if the environment should be terminated.
            truncated(MultiAgentDict): A bool values that indicates if the environment should be truncated.
            info (MultiAgentDict): The information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, *args, **kwargs):
        """
        Renders the environment.

        Return:
            rgb_images (np.ndarray or list): The images used to visualize the environment.
        """
        return NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the environment."""
        return NotImplementedError
