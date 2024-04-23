import numpy as np
from gym.spaces import Box
import time
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from xuance.environment.drones.customized.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from xuance.environment.drones.customized.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


class Drones_Env:
    def __init__(self, args):
        # import scenarios of gym-pybullet-drones
        self.env_id = args.env_id

        REGISTRY = {
            "CtrlAviary": CtrlAviary,
            "HoverAviary": HoverAviary,
            "VelocityAviary": VelocityAviary,
            "MultiHoverAviary": MultiHoverAviary,
            # you can add your customized scenarios here.
        }
        self.gui = args.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = args.sleep
        self.env_id = args.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["HoverAviary", "MultiHoverAviary"]:
            kwargs_env.update({'obs': ObservationType(args.obs_type),
                               'act': ActionType(args.act_type)})
        if self.env_id != "HoverAviary":
            kwargs_env.update({'num_drones': args.num_drones})
        self.env = REGISTRY[args.env_id](**kwargs_env)

        self._episode_step = 0
        self._episode_score = 0.0
        if self.env_id == "MultiHoverAviary":
            self.observation_space = self.env.observation_space
            self.observation_shape = self.env.observation_space.shape
            self.action_space = self.env.action_space
            self.action_shape = self.env.action_space.shape
        else:
            self.observation_space = self.space_reshape(self.env.observation_space)
            self.action_space = self.space_reshape(self.env.action_space)
        self.max_episode_steps = self.max_cycles = args.max_episode_steps

        self.n_agents = args.num_drones
        self.env_info = {
            "n_agents": self.n_agents,
            "obs_shape": self.env.observation_space.shape,
            "act_space": self.action_space,
            "state_shape": 20,
            "n_actions": self.env.action_space.shape[-1],
            "episode_limit": self.max_episode_steps,
        }

    def space_reshape(self, gym_space):
        low = gym_space.low.reshape(-1)
        high = gym_space.high.reshape(-1)
        shape_obs = (gym_space.shape[-1], )
        return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return np.zeros([2, 2, 2])

    def reset(self):
        obs, info = self.env.reset()
        info["episode_step"] = self._episode_step

        self._episode_step = 0
        if self.n_agents > 1:
            self._episode_score = np.zeros([self.n_agents, 1])
            obs_return = obs
        else:
            self._episode_score = 0.0
            obs_return = obs.reshape(-1)
        return obs_return, info

    def step(self, actions):
        if self.n_agents > 1:
            obs, reward, terminated, truncated, info = self.env.step(actions)
            obs_return = obs
            terminated = [terminated for _ in range(self.n_agents)]
        else:
            obs, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))
            obs_return = obs.reshape(-1)

        self._episode_step += 1
        self._episode_score += reward
        if self.n_agents > 1:
            truncated = [True for _ in range(self.n_agents)] if (self._episode_step >= self.max_episode_steps) else [False for _ in range(self.n_agents)]
        else:
            truncated = True if (self._episode_step >= self.max_episode_steps) else False
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards

        if self.gui:
            time.sleep(self.sleep)

        return obs_return, reward, terminated, truncated, info

    def get_agent_mask(self):
        return np.ones(self.n_agents, dtype=np.bool_)  # 1 means available

    def state(self):
        return np.zeros([20])
