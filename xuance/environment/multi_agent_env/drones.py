"""
gym-pybullet-drones
GitHub: https://github.com/utiasDSL/gym-pybullet-drones.git
Note: The version of Python should be >= 3.10.
"""
import numpy as np
from gym.spaces import Box
import time
from operator import itemgetter
from xuance.environment import RawMultiAgentEnv
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary as MultiHoverAviary_Official


class MultiHoverAviary(MultiHoverAviary_Official):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
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
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
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
        self.TARGET_POS = np.array([[0, 0, 1],
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [0, 0, 2],
                                    [0, 1, 2],
                                    [1, 0, 2],
                                    [2, 0, 1],
                                    [0, 2, 1],
                                    [2, 0, 2],
                                    [0, 2, 2], ])
        self.NUM_TARGETS = self.NUM_DRONES
        self.space_range_x = [-10.0, 10.0]
        self.space_range_y = [-10.0, 10.0]
        self.space_range_z = [0.02, 10.0]
        self.pose_limit = np.pi - 0.2

        ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        target_pos = self.TARGET_POS[:self.NUM_TARGETS].reshape(self.NUM_TARGETS, 1, 3)
        current_pos = states[:, :3].reshape(1, self.NUM_DRONES, 3)
        relative_pos = target_pos - current_pos
        distance_matrix = np.linalg.norm(relative_pos, axis=-1)
        reward_team = -distance_matrix.min(axis=-1, keepdims=True).sum()
        rewards = np.ones([self.NUM_DRONES, 1]) * reward_team

        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (max(abs(states[i][7]), abs(states[i][8])) > self.pose_limit) and (
                    z < self.space_range_z[0] + 0.05):  # the drone fulls down
                rewards[i] -= 10
            for j in range(self.NUM_DRONES):  # penalize collision with each other
                if i == j: continue
                distance_ij = np.linalg.norm(states[i, :3] - states[j, :3])
                if distance_ij < 0.1:
                    rewards[i] -= 10

        return rewards

        ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (max(abs(states[i][7]), abs(states[i][8])) > self.pose_limit) and (z < self.space_range_z[0] + 0.05):
                # The drone is too tilted
                return True

        return False

        ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (x < self.space_range_x[0]) or (x > self.space_range_x[1]) or (y < self.space_range_y[0]) or (
                    y > self.space_range_y[1]) or (z < self.space_range_z[0]) or (
                    z > self.space_range_z[1]):  # out of range
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False


REGISTRY = {
    "MultiHoverAviary": MultiHoverAviary,
    # you can add your customized scenarios here.
}


class Drones_MultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(Drones_MultiAgentEnv, self).__init__()
        # import scenarios of gym-pybullet-drones
        self.env_id = config.env_id
        self.gui = config.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = config.sleep
        self.env_id = config.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["MultiHoverAviary"]:
            kwargs_env.update({'num_drones': config.num_drones,
                               'obs': ObservationType(config.obs_type),
                               'act': ActionType(config.act_type)})
        self.env = REGISTRY[config.env_id](**kwargs_env)
        self.num_agents = config.num_drones
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.state_space = Box(-np.inf, np.inf, shape=[20, ])
        obs_shape_i = (self.env.observation_space.shape[-1],)
        act_shape_i = (self.env.action_space.shape[-1],)
        self.observation_space = {k: Box(-np.inf, np.inf, obs_shape_i) for k in self.agents}
        self.action_space = {k: Box(-np.inf, np.inf, act_shape_i) for k in self.agents}

        self.max_episode_steps = self.max_cycles = config.max_episode_steps
        self._episode_step = 0

    def space_reshape(self, gym_space):
        low = gym_space.low.reshape(-1)
        high = gym_space.high.reshape(-1)
        shape_obs = (gym_space.shape[-1],)
        return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return np.zeros([2, 2, 2])

    def reset(self):
        obs, info = self.env.reset()
        info["episode_step"] = self._episode_step
        self._episode_step = 0
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        return obs_dict, info

    def step(self, actions):
        actions_array = np.array(itemgetter(*self.agents)(actions))
        obs, reward, terminated, truncated, info = self.env.step(actions_array)
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        terminated_dict = {k: terminated for i, k in enumerate(self.agents)}
        rewrds_dict = {k: reward[i, 0] for i, k in enumerate(self.agents)}

        self._episode_step += 1
        truncated = True if (self._episode_step >= self.max_episode_steps) else False
        info["episode_step"] = self._episode_step  # current episode step

        if self.gui:
            time.sleep(self.sleep)

        return obs_dict, rewrds_dict, terminated_dict, truncated, info

    def agent_mask(self):
        return {agent: True for agent in self.agents}  # 1 means available

    def state(self):
        return self.state_space.sample()

    def avail_actions(self):
        return
