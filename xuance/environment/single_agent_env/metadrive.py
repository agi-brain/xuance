from xuance.environment import RawEnvironment
from metadrive.envs.metadrive_env import MetaDriveEnv


class MetaDrive_Env(RawEnvironment):
    """
    The raw environment of MetaDrive in XuanCe.
    Parameters:
        configs: the configurations of the environment.
    """
    def __init__(self, configs):
        super(MetaDrive_Env, self).__init__()
        self.env_id = configs.env_id
        configs.env_config['use_render'] = configs.render
        self.env = MetaDriveEnv(config=configs.env_config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.episode_lengths

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render()

    def close(self):
        self.env.close()
