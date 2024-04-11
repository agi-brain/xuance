import numpy as np
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
        # self.TARGET_POS = self.INIT_XYZS + np.array([[0, 0, 1 / (i + 1)] for i in range(num_drones)])
        self.TARGET_POS = np.array([[0, 0, 1],
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [0, 0, 2],
                                    [0, 1, 2],
                                    [1, 0, 2],
                                    [2, 0, 1],
                                    [0, 2, 1],
                                    [2, 0, 2],
                                    [0, 2, 2],])
        self.permutation = np.random.permutation(num_drones)
        self.space_range_x = [-20.0, 20.0]
        self.space_range_y = [-20.0, 20.0]
        self.space_range_z = [0.05, 5.0]
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
        rewards = np.zeros([self.NUM_DRONES, 1])
        for i in range(self.NUM_DRONES):
            rewards[i, 0] = max(0, (1 - np.linalg.norm(self.TARGET_POS[i] - states[i][0:3])) * 20)
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
            if (x < self.space_range_x[0]) or (x > self.space_range_x[1]):
                return True
            if (y < self.space_range_y[0]) or (y > self.space_range_y[1]):
                return True
            if (z < self.space_range_z[0]) or (z > self.space_range_z[1]):  # Out of height
                return True
            if (max(abs(states[i][7]), abs(states[i][8])) > self.pose_limit) and (z < self.space_range_z[0] + 0.05):
                # The drone is too tilted
                return True
            if np.linalg.norm(self.TARGET_POS[i] - states[i][0:3]) < .0001:
                return True
        return False

        ################################################################################
