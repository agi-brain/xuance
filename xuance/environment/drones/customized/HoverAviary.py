import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary as HoverAviary_Official


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
