import gymnasium
try:
    import gym as gym_legacy  # gym-platform is registered against legacy gym.
    import gym_platform
except ImportError:
    pass


def _to_gymnasium_space(space):
    """Convert a legacy gym space to its gymnasium equivalent (recursively).

    gym-platform exposes legacy gym spaces, but the rest of XuanCe expects
    gymnasium spaces (e.g. gymnasium.spaces.Tuple asserts its members are
    gymnasium spaces), so the wrapper converts them at the boundary.
    """
    name = type(space).__name__
    if name == "Box":
        return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if name == "Discrete":
        return gymnasium.spaces.Discrete(space.n)
    if name == "Tuple":
        return gymnasium.spaces.Tuple(tuple(_to_gymnasium_space(s) for s in space.spaces))
    raise NotImplementedError(f"Unsupported legacy gym space type: {name}")


class PlatformEnv:
    """
    The wrapper of gym-platform environment.
    Environment link: https://github.com/cycraig/gym-platform.git

    Args:
        config: the configurations for the environment.
    """
    def __init__(self, config):
        super(PlatformEnv, self).__init__()
        self.env_id = config.env_id
        self.render_mode = config.render_mode
        env = gym_legacy.make(self.env_id)
        self.env = env.unwrapped
        self.env.seed(config.env_seed)  # legacy gym API: seed() then reset().
        self.env.reset()
        self.num_envs = 1

        self.observation_space = _to_gymnasium_space(self.env.observation_space)
        self.action_space = _to_gymnasium_space(self.env.action_space)
        self.max_episode_steps = config.max_episode_steps

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self):
        """Reset the environment."""
        return self.env.reset()

    def step(self, action):
        """Execute the actions and get next observations, rewards, and other information."""
        return self.env.step(action)
