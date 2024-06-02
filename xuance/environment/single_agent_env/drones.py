"""
gym-pybullet-drones
GitHub: https://github.com/utiasDSL/gym-pybullet-drones.git
Note: The version of Python should be >= 3.10.
"""
import time
import numpy as np
from gym.spaces import Box
from xuance.environment import RawEnvironment
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary as HoverAviary_Official
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class HoverAviary(HoverAviary_Official):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8  # ?
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.TARGET_POS = np.array([0, 0, 1])
        self.space_range = [2.0, 2.0]
        self.pose_limit = np.pi - 0.2
        self.height_limit = [0.05, 5.0]

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        reward = max(0, (1 - np.linalg.norm(self.TARGET_POS - state[0:3])) * 20)
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.space_range[0]) or (abs(state[1]) > self.space_range[1]):  # Out of range
            return True
        if (state[2] > self.height_limit[1]) or (state[2] < self.height_limit[0]):  # Out of height
            return True
        if (abs(state[7]) > self.pose_limit or abs(state[8]) > self.pose_limit) and (
                state[2] < self.height_limit[0]):  # Truncate when the drone is too tilted
            return True
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .0001:
            return True
        else:
            return False


REGISTRY = {
    "CtrlAviary": CtrlAviary,
    "HoverAviary": HoverAviary,
    "VelocityAviary": VelocityAviary,
    # you can add your customized scenarios here.
}


class Drone_Env(RawEnvironment):
    def __init__(self, config):
        super(Drone_Env, self).__init__()
        # import scenarios of gym-pybullet-drones
        self.env_id = config.env_id

        self.gui = config.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = config.sleep
        self.env_id = config.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["HoverAviary"]:
            kwargs_env.update({'obs': ObservationType(config.obs_type),
                               'act': ActionType(config.act_type)})
        if self.env_id != "HoverAviary":
            kwargs_env.update({'num_drones': config.num_drones})
        self.env = REGISTRY[config.env_id](**kwargs_env)

        self._episode_step = 0
        self.observation_space = self.space_reshape(self.env.observation_space)
        self.action_space = self.space_reshape(self.env.action_space)
        self.max_episode_steps = config.max_episode_steps

    def space_reshape(self, gym_space):
        low = gym_space.low.reshape(-1)
        high = gym_space.high.reshape(-1)
        shape_obs = (gym_space.shape[-1], )
        return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0
        obs_return = obs.reshape(-1)
        return obs_return, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))
        obs_return = obs.reshape(-1)
        self._episode_step += 1
        truncated = True if (self._episode_step >= self.max_episode_steps) else False
        if self.gui:
            time.sleep(self.sleep)
        return obs_return, reward, terminated, truncated, info



