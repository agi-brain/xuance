"""
    2 agents, 3 landmarks of different colors. Each agent wants to 
    get to their target landmark, which is known only by other agent. 
    Reward is collective. So agents have to learn to communicate 
    the goal of the other agent, and navigate to their landmark. 
    This is the same as the simple_speaker_listener scenario 
    where both agents are simultaneous speakers and listeners.
    Same as simple_reference, except one agent is the ‘speaker’ (gray) 
    that does not move (observes goal of other agent), and other 
    agent is the listener (cannot speak, but must navigate to 
    correct landmark).
"""
import argparse
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 3
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.num_agents = args.num_agents  # 2
        assert world.num_agents == 2, "only 2 agents is supported, check the config.py."
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.current_time_step = 0
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array(
            [0.45, 0.45, 0.45]
        )
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)


if __name__ == "__main__":
    from multiagent.environment import MultiAgentOrigEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents: int = 2
            self.num_landmarks: int = 3
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
        done_callback=scenario.done if hasattr(scenario, "done") else None,
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
