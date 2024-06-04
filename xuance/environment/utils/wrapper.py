from typing import Optional, Tuple, SupportsFloat
from gym import spaces


class XuanCeEnvWrapper:
    """
    Wraps an environment for single-agent system that can run in XuanCe.
    """

    def __init__(self, env, **kwargs):
        super(XuanCeEnvWrapper, self).__init__()
        self.env = env
        self._action_space: Optional[spaces.Space] = None
        self._observation_space: Optional[spaces.Space] = None
        self._reward_range: Optional[Tuple[SupportsFloat, SupportsFloat]] = None
        self._metadata: Optional[dict] = None
        self._max_episode_steps: Optional[int] = None
        self._episode_step = 0
        self._episode_score = 0.0

    @property
    def action_space(self):
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

    def step(self, action):
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
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

    def render(self, *args, **kwargs):
        """Renders the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env


class XuanCeMultiAgentEnvWrapper(XuanCeEnvWrapper):
    """
    Wraps an environment for multi-agent system that can run in XuanCe.
    """

    def __init__(self, env, **kwargs):
        super(XuanCeMultiAgentEnvWrapper, self).__init__(env, **kwargs)
        self._env_info: Optional[dict] = None
        self._state_space: Optional[spaces.Space] = None
        self.agents = self.env.agents  # e.g., ['red_0', 'red_1', 'blue_0', 'blue_1'].
        self.num_agents = self.env.num_agents  # Number of all agents, e.g., 4.
        self.teams_info = {  # Information of teams.
            "names": ['red', 'blue'],  # should be consistent with the name of agents.
            "num_teams": 2,
            "agents_in_team": [["red_0", 'red_1'], ['blue_0', 'blue_1']]
        }
        self._episode_score = {agent: 0.0 for agent in self.agents}

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """Resets the environment with kwargs."""
        try:
            obs, info = self.env.reset(**kwargs)
        except:
            obs = self.env.reset(**kwargs)
            info = {}
        self._episode_step = 0
        self._episode_score = {agent: 0.0 for agent in self.agents}
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        info["agent_mask"] = self.agent_mask
        info["avail_actions"] = self.avail_actions
        info["state"] = self.state
        return obs, info

    def step(self, action):
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
        for agent in self.agents:
            self._episode_score[agent] += reward[agent]
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        info["agent_mask"] = self.agent_mask
        info["avail_actions"] = self.avail_actions
        info["state"] = self.state
        return observation, reward, terminated, truncated, info

    @property
    def env_info(self) -> Optional[dict]:
        """Returns the information of the environment."""
        if self._env_info is None:
            return self.env.env_info
        return self._env_info

    @env_info.setter
    def env_info(self, info: {}):
        """Sets the action space"""
        self._env_info = info

    @property
    def state_space(self) -> spaces.Space:
        """Returns the global state space of the environment."""
        if self._state_space is None:
            return self.env.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space: spaces.Space):
        """Sets the global state space."""
        self._state_space = space

    @property
    def state(self):
        """Returns global states in the multi-agent environment."""
        return self.env.state()

    @property
    def agent_mask(self):
        """Returns mask variables to mark alive agents in multi-agent environment."""
        return self.env.agent_mask()

    @property
    def avail_actions(self):
        """Returns mask variables to mark available actions for each agent."""
        return self.env.avail_actions()
