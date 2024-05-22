import numpy as np
from baselines.mpnn.mpe.mape.multiagent.core import World, Agent, Landmark
from baselines.mpnn.mpe.mape.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class Scenario(BaseScenario):
    def __init__(self, num_agents=3, dist_threshold=0.1, arena_size=1, identity_size=0):
        self.num_agents = num_agents
        self.rewards = np.zeros(self.num_agents)
        self.temp_done = False
        self.dist_threshold = dist_threshold
        self.arena_size = arena_size
        self.identity_size = identity_size

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = self.num_agents
        num_landmarks = num_agents
        world.collaborative = False
        # add agents
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.adversary = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        world.dists = []
        world.dist_thres = self.dist_threshold
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

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

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(
                -self.arena_size, self.arena_size, world.dim_p
            )
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.steps = 0
        world.dists = []

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        if agent.iden == 0:  # compute this only once when called with the first agent
            # each column represents distance of all agents from the respective landmark
            world.dists = np.array(
                [
                    [
                        np.linalg.norm(a.state.p_pos - l.state.p_pos)
                        for l in world.landmarks
                    ]
                    for a in world.agents
                ]
            )
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = self._bipartite_min_dists(world.dists)
            # the reward is normalized by the number of agents
            joint_reward = np.clip(-np.mean(self.min_dists), -15, 15)
            self.rewards = np.full(self.num_agents, joint_reward)
            world.min_dists = self.min_dists
        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
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
        self.is_success = np.all(self.min_dists < world.dist_thres)
        return condition1 or self.is_success

    def info(self, agent, world):
        info = {
            "is_success": self.is_success,
            "world_steps": world.steps,
            "reward": self.rewards.mean(),
            "dists": self.min_dists.mean(),
        }
        return info
