import numpy as np
from gym.spaces import Box


class Drones_Env:
    def __init__(self, args):
        # import scenarios of gym-pybullet-drones
        self.env_id = args.env_id
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
        from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
        from gym_pybullet_drones.utils.enums import ObservationType, ActionType
        REGISTRY = {
            "CtrlAviary": CtrlAviary,
            "HoverAviary": HoverAviary,
            "VelocityAviary": VelocityAviary,
            "MultiHoverAviary": MultiHoverAviary,
        }
        self.env_id = args.env_id

        kwargs_env = {'gui': args.render}
        if self.env_id in ["HoverAviary", "MultiHoverAviary"]:
            kwargs_env.update({'obs': ObservationType(args.obs_type),
                               'act': ActionType(args.act_type)})
        self.env = REGISTRY[args.env_id](**kwargs_env)
        self._episode_step = 0
        self._episode_score = 0.0
        self.observation_space = self.space_reshape(self.env.observation_space)
        self.action_space = self.space_reshape(self.env.action_space)
        self.max_episode_steps = self.max_cycles = args.max_episode_steps

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
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs.reshape(-1), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards

        truncated = True if (self._episode_step >= self.max_episode_steps) else False

        return observation.reshape(-1), reward, terminated, truncated, info


