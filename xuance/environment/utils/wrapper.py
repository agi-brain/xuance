from typing import Optional, Union, Tuple, List, SupportsFloat, SupportsInt
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame


class XuanCeEnvWrapprer:
    """
    Wraps an environment for single-agent system that can run in XuanCe.
    """
    def __new__(cls, *args, **kwargs):
        if cls is XuanCeEnvWrapprer:
            raise TypeError("Type XuanCeEnvWrapprer cannot be instantiated; It can be used only as a base class")
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, env, **kwargs):
        super(XuanCeEnvWrapprer, self).__init__()
        self.env = env
        self._action_space: Optional[spaces.Space] = None
        self._observation_space: Optional[spaces.Space] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[dict] = None
        self._max_episode_steps: Optional[int] = None

    @property
    def action_space(self) -> spaces.Space[ActType]:
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        """Sets the action space"""
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        """Sets the observation space."""
        self._observation_space = space

    @property
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """Return the reward range of the environment."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[SupportsFloat, SupportsFloat]):
        """Sets reward range."""
        self._reward_range = value

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Sets metadata"""
        self._metadata = value

    @property
    def max_episode_steps(self) -> int:
        """Returns the maximum of episode steps."""
        if self._max_episode_steps is None:
            return self.env.max_episode_steps
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        """Sets the maximum of episode steps"""
        self._max_episode_steps = value

    @property
    def render_mode(self) -> Optional[str]:
        """Returns the environment render_mode."""
        return self.env.render_mode

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        """Resets the environment with kwargs."""
        try:
            obs, info = self.env.reset(**kwargs)
        except:
            obs = self.env.reset(**kwargs)
            info = {}
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def render(
            self, *args, **kwargs
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped


class XuanCeMultiAgentEnvWrapprer:
    """
    Wraps an environment for multi-agent system that can run in XuanCe.
    """
    def __new__(cls, *args, **kwargs):
        if cls is XuanCeEnvWrapprer:
            raise TypeError("Type XuanCeEnvWrapprer cannot be instantiated; It can be used only as a base class")
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, env, **kwargs):
        super(XuanCeMultiAgentEnvWrapprer, self).__init__()
        self.env = env
        self._action_space: Optional[spaces.Space] = None
        self._observation_space: Optional[spaces.Space] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[dict] = None
        self._max_episode_steps: Optional[int] = None

    @property
    def action_space(self) -> spaces.Space[ActType]:
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        """Sets the action space"""
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        """Sets the observation space."""
        self._observation_space = space

    @property
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """Return the reward range of the environment."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[SupportsFloat, SupportsFloat]):
        """Sets reward range."""
        self._reward_range = value

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Sets metadata"""
        self._metadata = value

    @property
    def max_episode_steps(self) -> int:
        """Returns the maximum of episode steps."""
        if self._max_episode_steps is None:
            return self.env.max_episode_steps
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        """Sets the maximum of episode steps"""
        self._max_episode_steps = value

    @property
    def render_mode(self) -> Optional[str]:
        """Returns the environment render_mode."""
        return self.env.render_mode

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        """Resets the environment with kwargs."""
        try:
            obs, info = self.env.reset(**kwargs)
        except:
            obs = self.env.reset(**kwargs)
            info = {}
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def render(
            self, *args, **kwargs
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

