import numpy as np
from baselines.mpnn.mpe.mape.multiagent.core import World, Agent, Landmark
from baselines.mpnn.mpe.mape.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.arena_size = arena_size
        self.num_agents = num_agents
        self.total_sep = 1.25 * self.arena_size
        self.ideal_sep = self.total_sep / (num_agents - 1)
        self.dist_thres = 0.05
        self.identity_size = identity_size

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = 2
        world.collaborative = False

        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.adversary = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02

        # make initial conditions
        self.reset_world(world)
        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        # colors = [np.array([0,0,0.1]), np.array([0,1,0]), np.array([0,0,1]), np.array([1,1,0]), np.array([1,0,0])]
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # agent.color = colors[i]

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(
                -self.arena_size, self.arena_size, world.dim_p
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.random.uniform(
            -0.25 * self.arena_size, 0.25 * self.arena_size, world.dim_p
        )
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        theta = np.random.uniform(0, 2 * np.pi)
        loc = world.landmarks[0].state.p_pos + self.total_sep * np.array(
            [np.cos(theta), np.sin(theta)]
        )
        # find a suitable theta such that landmark 1 is within the bounds
        while not (abs(loc[0]) < self.arena_size and abs(loc[1]) < self.arena_size):
            theta += np.radians(5)
            loc = world.landmarks[0].state.p_pos + self.total_sep * np.array(
                [np.cos(theta), np.sin(theta)]
            )

        world.landmarks[1].state.p_pos = loc
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        self.expected_positions = [
            world.landmarks[0].state.p_pos
            + i * self.ideal_sep * np.array([np.cos(theta), np.sin(theta)])
            for i in range(len(world.agents))
        ]

        world.steps = 0
        world.dists = []

    def reward(self, agent, world):
        if agent.iden == 0:
            dists = np.array(
                [
                    [
                        np.linalg.norm(a.state.p_pos - pos)
                        for pos in self.expected_positions
                    ]
                    for a in world.agents
                ]
            )
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.delta_dists = self._bipartite_min_dists(dists)
            world.dists = self.delta_dists

            total_penalty = np.mean(np.clip(self.delta_dists, 0, 2))
            self.joint_reward = -total_penalty

        return self.joint_reward

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame
        entity_pos = [
            entity.state.p_pos - agent.state.p_pos for entity in world.landmarks
        ]
        default_obs = np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos
        )
        if self.identity_size != 0:
            identified_obs = np.append(
                np.eye(self.identity_size)[agent.iden], default_obs
            )
            return identified_obs
        return default_obs

    def done(self, agent, world):
        condition1 = world.steps >= world.max_steps_episode
        self.is_success = np.all(self.delta_dists < self.dist_thres)
        return condition1 or self.is_success

    def info(self, agent, world):
        return {
            "is_success": self.is_success,
            "world_steps": world.steps,
            "reward": self.joint_reward,
            "dists": self.delta_dists.mean(),
        }
