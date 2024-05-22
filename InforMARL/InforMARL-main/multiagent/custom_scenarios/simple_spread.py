"""
    N agents, N landmarks. 
    Agents are rewarded based on how far any agent is from each landmark. 
    Agents are penalized if they collide with other agents. 
    So, agents have to learn to cover all the landmarks while 
    avoiding collisions.
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace) -> World:
        world = World()
        world.world_length = args.episode_length
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 2
        self.num_agents = args.num_agents
        num_landmarks = self.num_agents
        world.collaborative = args.collaborative
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        world.current_time_step = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent: Agent, world: World) -> Tuple:
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent: Agent, world: World) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark,
        # penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )

    def done(self, agent: Agent, world: World) -> bool:
        # done is False if done_callback is not passed to
        # environment.MultiAgentEnv
        # This is same as original version
        # Check `_get_done()` in environment.MultiAgentEnv
        return False


if __name__ == "__main__":
    from multiagent.environment import MultiAgentOrigEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents: int = 3
            self.num_obstacles: int = 3
            self.collaborative: bool = False
            self.max_speed: Optional[float] = 2
            self.collision_rew: float = 5
            self.goal_rew: float = 5
            self.min_dist_thresh: float = 0.1
            self.use_dones: bool = False
            self.episode_length: int = 25

    args = Args()

    scenario = Scenario()

    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentOrigEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        # info_callback=scenario.info_callback,
        done_callback=scenario.done,
        shared_viewer=False,
    )
    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    stp = 0
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        # render all agent views
        env.render()
        stp += 1
