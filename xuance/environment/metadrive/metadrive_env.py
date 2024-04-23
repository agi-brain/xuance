from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np


class MetaDrive_Env:
    def __init__(self, args):
        self.env_id = args.env_id
        args.env_config['use_render'] = args.render
        self.env = MetaDriveEnv(config=args.env_config)

        self._episode_step = 0  # The count of steps for current episode.
        self._episode_score = 0.0  # The cumulated rewards for current episode.
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.episode_lengths

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return np.zeros([2, 2, 2])

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0  # The count of steps for current episode.
        self._episode_score = 0.0  # The cumulated rewards for current episode.
        info["episode_step"] = self._episode_step
        return obs, info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info
