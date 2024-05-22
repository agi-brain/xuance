"""
    2 agents, 3 landmarks of different colors. Each agent wants to 
    get to their target landmark, which is known only by other agent. 
    Reward is collective. So agents have to learn to communicate 
    the goal of the other agent, and navigate to their landmark. 
    This is the same as the simple_speaker_listener scenario 
    where both agents are simultaneous speakers and listeners.
"""
import argparse
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace):
        world = World()
        # set any world properties first
        world.world_length = args.episode_length
        world.current_time_step = 0
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.num_agents = args.num_agents  # 2
        assert world.num_agents == 2, "only 2 agents is supported, check the config.py."
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.num_landmarks = args.num_landmarks  # 3
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.current_time_step = 0
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2  # np.exp(-dist2)

    def observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

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
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)


if __name__ == "__main__":
    from multiagent.environment import MultiAgentOrigEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents: int = 2
            self.episode_length: int = 25
            self.num_landmarks: int = 3

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
