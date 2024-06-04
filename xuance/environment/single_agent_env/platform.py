import gym
import gym_platform


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
        env = gym.make(self.env_id, max_episode_steps=config.max_episode_steps)
        self.env = env.unwrapped
        self.num_envs = 1

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
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
