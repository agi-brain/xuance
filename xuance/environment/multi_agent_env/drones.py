"""
gym-pybullet-drones
GitHub: https://github.com/utiasDSL/gym-pybullet-drones.git
Note: The version of Python should be >= 3.10.
"""
import numpy as np
from gym.spaces import Box
import time
from xuance.environment import RawMultiAgentEnv
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary as MultiHoverAviary_Official


class Drones_MultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, args):
        super(Drones_MultiAgentEnv, self).__init__()
        # import scenarios of gym-pybullet-drones
        self.env_id = args.env_id
        REGISTRY = {
            "MultiHoverAviary": MultiHoverAviary,
            # you can add your customized scenarios here.
        }
        self.gui = args.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = args.sleep
        self.env_id = args.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["MultiHoverAviary"]:
            kwargs_env.update({'num_drones': args.num_drones,
                              'obs': ObservationType(args.obs_type),
                               'act': ActionType(args.act_type)})
        self.env = REGISTRY[args.env_id](**kwargs_env)

        self._episode_step = 0
        if self.env_id == "MultiHoverAviary":
            self.observation_space = self.env.observation_space
            self.observation_shape = self.env.observation_space.shape
            self.action_space = self.env.action_space
            self.action_shape = self.env.action_space.shape
        else:
            self.observation_space = self.space_reshape(self.env.observation_space)
            self.action_space = self.space_reshape(self.env.action_space)
        self.max_episode_steps = self.max_cycles = args.max_episode_steps

        self.num_agents = args.num_drones
        self.env_info = {
            "n_agents": self.num_agents,
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
        obs_return = obs
        return obs_return, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs_return = obs
        terminated = [terminated for _ in range(self.num_agents)]

        self._episode_step += 1
        truncated = True if (self._episode_step >= self.max_episode_steps) else False
        info["episode_step"] = self._episode_step  # current episode step

        if self.gui:
            time.sleep(self.sleep)

        return obs_return, reward, terminated, truncated, info

    def get_agent_mask(self):
        return np.ones(self.num_agents, dtype=np.bool_)  # 1 means available

    def get_state(self):
        return np.zeros([20])


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
