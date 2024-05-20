from typing import Optional
from abc import ABC, abstractmethod
from gym import spaces


class RawEnvironment(ABC):
    """A base class for new environment."""
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
    def step(self, action):
        """Steps through the environment with action."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        raise NotImplementedError

    @abstractmethod
    def render(self, *args, **kwargs):
        """Renders the environment."""
        return NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the environment."""
        return NotImplementedError
