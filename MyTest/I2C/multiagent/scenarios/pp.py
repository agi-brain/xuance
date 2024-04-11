import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 7
        num_preys = 3
        world.collaborative = False
        world.discrete_action = True
        world.num_agents_obs = 3
        world.num_preys_obs = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
            agent.accel = 5.0
        # add landmarks
        world.preys = [Agent() for i in range(num_preys)]
        for i, prey in enumerate(world.preys):
            prey.name = 'prey %d' % i
            prey.collide = False
            prey.movable = True
            prey.silent = True
            prey.size = 0.05
            prey.accel = 7.0
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, prey in enumerate(world.preys):
            prey.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, prey in enumerate(world.preys):
            prey.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            if i != 0:
                for j in range(i): 
                    while True:
                        if np.sqrt(np.sum(np.square(prey.state.p_pos - world.preys[j].state.p_pos)))>0.00:
                            break
                        else: prey.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            prey.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_preys = 0
        min_dists = 0
        for p in world.preys:
            collision_dist = agent.size + p.size
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - p.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < collision_dist:
                occupied_preys += 1
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        collisions += 1
        return (rew, collisions/2, min_dists, occupied_preys)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_dist = agent1.size + agent2.size
        return True if dist < collision_dist else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # local reward
        #dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        #rew = rew - min(dists)
        # global reward
        for l in world.preys:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 0.5
        return rew

    def observation(self, agent, world):
        # get positions of predefined preys
        entity_pos = []
        dis_lm_n = []
        num_preys_obs = world.num_preys_obs
        for entity in world.preys:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            dis_lm_n.append(np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos))))
        sort_index = sorted(range(len(dis_lm_n)), key=lambda k: dis_lm_n[k])
        near_lm_pos = [entity_pos[sort_index[i]] for i in range(num_preys_obs)]
        # get positions of predefined agents
        other_pos = []
        dis_agent_n = []
        num_agents_obs = world.num_agents_obs
        for other in world.agents:
            if other is agent: continue
            dis_agent_n.append(np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos))))
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        sort_index = sorted(range(len(dis_agent_n)), key=lambda k: dis_agent_n[k])
        near_agent_pos = [other_pos[sort_index[i]] for i in range(num_agents_obs)]
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + near_lm_pos + near_agent_pos)


