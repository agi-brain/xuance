import numpy as np
import rvo2

import matplotlib.pyplot as plt


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class RVOPolicy:
    def __init__(self):
        sensing_horizon = np.inf
        max_num_agents_in_env = 19
        rvo_time_horizon = 2.5  # NOTE: bjorn used 1.0 in training for corl19
        self.rvo_collab_coeff = 0.5
        self.rvo_anti_collab_t = 1.0
        self.dt = 0.1
        ########################################################################
        neighbor_dist = sensing_horizon
        max_neighbors = max_num_agents_in_env

        self.has_fixed_speed = False
        self.heading_noise = False

        self.max_delta_heading = np.pi / 6

        # TODO share this parameter with environment
        # Initialize RVO simulator
        self.sim = rvo2.PyRVOSimulator(
            timeStep=self.dt,
            neighborDist=neighbor_dist,
            maxNeighbors=max_neighbors,
            timeHorizon=rvo_time_horizon,
            timeHorizonObst=rvo_time_horizon,
            radius=0.0,
            maxSpeed=0.0,
        )

        self.is_init = False

        self.use_non_coop_policy = True

    def init(self):
        state_dim = 2
        self.pos_agents = np.empty((self.n_agents, state_dim))
        self.vel_agents = np.empty((self.n_agents, state_dim))
        self.goal_agents = np.empty((self.n_agents, state_dim))
        self.pref_vel_agents = np.empty((self.n_agents, state_dim))
        self.pref_speed_agents = np.empty((self.n_agents))

        self.rvo_agents = [None] * self.n_agents

        # Init simulation
        for a in range(self.n_agents):
            self.rvo_agents[a] = self.sim.addAgent((0, 0))

        self.is_init = True

    # NOTE: Right now just supports multi-discrete actions
    # actions: [None, ←, →, ↓, ↑, comm1, comm2]
    def convert_to_action(self, delta_pos):
        action = np.zeros(7)
        # if the required movement is not much, then `no_action`
        if np.linalg.norm(delta_pos) <= self.goal_threshold:
            action[0] = 1
        else:
            angle = np.deg2rad(np.arctan2(delta_pos[1], delta_pos[0]))
            # activate right action
            if angle < 67.5 and angle >= -67.5:
                action[2] = 1
            # activate up action
            if angle < 157.5 and angle >= 22.5:
                action[4] = 1
            # activate down action
            if angle >= -157.5 and angle < -22.5:
                action[3] = 1
            # activate left action
            if angle >= 112.5 or angle < -112.5:
                action[1] = 1
        return action

    def find_next_action(self, world, agents, agent_index):
        # Initialize vectors on first call to infer number of agents
        if not self.is_init:
            self.n_agents = len(agents)
            self.init()

        # Share all agent positions and preferred velocities from environment with RVO simulator
        for a in range(self.n_agents):
            # Copy current agent positions, goal and preferred speeds into np arrays
            self.pos_agents[a, :] = agents[a].state.p_pos
            self.goal_agents[a, :] = world.get_entity(
                "landmark", agents[a].id
            ).state.p_pos
            self.vel_agents[a, :] = agents[a].state.p_vel
            self.pref_speed_agents[a] = agents[a].pref_speed

            # Calculate preferred velocity
            # Assumes non RVO agents are acting like RVO agents
            self.pref_vel_agents[a, :] = self.goal_agents[a, :] - self.pos_agents[a, :]
            self.pref_vel_agents[a, :] = (
                self.pref_speed_agents[a]
                / np.linalg.norm(self.pref_vel_agents[a, :])
                * self.pref_vel_agents[a, :]
            )

            # Set agent positions and velocities in RVO simulator
            self.sim.setAgentMaxSpeed(self.rvo_agents[a], agents[a].pref_speed)
            self.sim.setAgentRadius(self.rvo_agents[a], (1 + 5e-2) * agents[a].radius)
            self.sim.setAgentPosition(self.rvo_agents[a], tuple(self.pos_agents[a, :]))
            self.sim.setAgentVelocity(self.rvo_agents[a], tuple(self.vel_agents[a, :]))
            self.sim.setAgentPrefVelocity(
                self.rvo_agents[a], tuple(self.pref_vel_agents[a, :])
            )

        # Set ego agent's collaborativity
        if self.rvo_collab_coeff < 0:
            # agent is anti-collaborative ==> every X seconds,
            # it chooses btwn non-coop and adversarial, where the PMF of
            # which policy to run is defined by abs(collab_coeff)\in(0,1].

            # if a certain freq, randomly select btwn use non coop policy vs. rvo
            if (
                round(agents[agent_index].t % self.rvo_anti_collab_t, 3) < self.dt
                or round(
                    self.rvo_anti_collab_t
                    - agents[agent_index].t % self.rvo_anti_collab_t,
                    3,
                )
                < self.dt
            ):
                self.use_non_coop_policy = np.random.choice(
                    [True, False],
                    p=[1 - abs(self.rvo_collabb_coeff), abs(self.rvo_collabb_coeff)],
                )
            if self.use_non_coop_policy:
                self.sim.setAgentCollabCoeff(self.rvo_agents[agent_index], 0.0)
            else:
                self.sim.setAgentCollabCoeff(
                    self.rvo_agents[agent_index], self.rvo_collabb_coeff
                )
        else:
            self.sim.setAgentCollabCoeff(
                self.rvo_agents[agent_index], self.rvo_collabb_coeff
            )

        # Execute one step in the RVO simulator
        self.sim.doStep()

        # Calculate desired change of heading
        self.new_rvo_pos = self.sim.getAgentPosition(self.rvo_agents[agent_index])[:]
        ########## process new pos to speed and angle ##########
        deltaPos = self.new_rvo_pos - self.pos_agents[agent_index, :]
        action = self.convert_to_action(delta_pos=deltaPos)
        # p1 = deltaPos
        # p2 = np.array([1,0]) # Angle zero is parallel to x-axis
        # ang1 = np.arctan2(*p1[::-1])
        # ang2 = np.arctan2(*p2[::-1])
        # new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
        # delta_heading = wrap(new_heading_global_frame - agents[agent_index].heading_global_frame)

        # # Calculate desired speed
        # pref_speed = 1/self.dt * np.linalg.norm(deltaPos)

        # # Limit the turning rate: stop and turn in place if exceeds
        # if abs(delta_heading) > self.max_delta_heading:
        #     delta_heading = np.sign(delta_heading)*self.max_delta_heading
        #     pref_speed = 0.

        # # Ignore speed
        # if self.has_fixed_speed:
        #     pref_speed = self.max_speed

        # # Add noise
        # if self.heading_noise:
        #     delta_heading = delta_heading + np.random.normal(0,0.5)

        # action = np.array([pref_speed, delta_heading])
        return action
