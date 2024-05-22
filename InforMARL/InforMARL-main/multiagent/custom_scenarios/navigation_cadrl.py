"""
    Navigation for `n` agents to `n` goals from random initial positions
    With random obstacles added in the environment
    Each agent is destined to get to its own goal unlike
    `simple_spread.py` where any agent can get to any goal (check `reward()`)
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
        """
        Parameters in args
        ––––––––––––––––––
        • num_agents: int
            Number of agents in the environment
            NOTE: this is equal to the number of goal positions
        • num_obstacles: int
            Number of num_obstacles obstacles
        • collaborative: bool
            If True then reward for all agents is sum(reward_i)
            If False then reward for each agent is what it gets individually
        • max_speed: Optional[float]
            Maximum speed for agents
            NOTE: Even if this is None, the max speed achieved in discrete
            action space is 2, so might as well put it as 2 in experiments
            TODO: make list for this and add this in the state
        • collision_rew: float
            The reward to be negated for collisions with other agents and
            obstacles
        • goal_rew: float
            The reward to be added if agent reaches the goal
        • min_dist_thresh: float
            The minimum distance threshold to classify whether agent has
            reached the goal or not
        • use_dones: bool
            Whether we want to use the 'done=True' when agent has reached
            the goal or just return False like the `simple.py` or
            `simple_spread.py`
        • episode_length: int
            Episode length after which environment is technically reset()
            This determines when `done=True` for done_callback
        • obs_type: str
            Choices: ['global', 'local', 'nbd']
            Whether we want include the [pos, vel] of just the neighbourhood
            instead of all entities in the environment for the observation
        ____________________________________________________________________
        If `obs_type== global` then the observation is the concatenation of
            [pos, vel] of all entities in the environment
        If `obs_type== local` then the observation is the [pos, vel] of the
            agent
        If `obs_type == ndb_obs` then each agent's observation is just the
            [pos, vel] of  neighbouring `n` entities, provided that they are
            within `max_edge_dist` of each other. If there are less than `n`
            entities within `max_edge_dist` of each other, then pad with zeros.
        ____________________________________________________________________
        • max_edge_dist: float
            Maximum distance for an entity to be considered as neighbour
        • ndb_entities: int
            Number of neighbouring entities to consider in the environment
        • use_comm: bool
            Whether we want to use communication or not
        """
        # pull params from args
        self.num_agents = args.num_agents
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.max_speed = args.max_speed
        self.collision_rew = args.collision_rew
        self.goal_rew = args.goal_rew
        self.min_dist_thresh = args.min_dist_thresh
        self.use_dones = args.use_dones
        self.episode_length = args.episode_length
        self.obs_type = "local"
        self.max_edge_dist = args.max_edge_dist
        # agent specific params, probably not needed:
        # self.px = args.px
        # self.py = args.py
        # self.vx = args.vx
        # self.vy = args.vy
        # self.radius = args.radius
        # self.pgx = args.pgx
        # self.pgy = args.pgy
        # self.v_pref = args.v_pref
        # self.theta = args.theta
        # self.kinematic = args.kinematic

        # self.num_nbd_entities = args.num_nbd_entities
        self.use_comm = False
        ####################
        world = World()
        world.world_length = args.episode_length
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # set any world properties
        world.dim_c = 2
        num_landmarks = self.num_agents  # no. of goals equal to no. of agents
        world.collaborative = args.collaborative

        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            # NOTE not changing size of agent because of some edge cases;
            # TODO have to change this later
            # agent.size = 0.15
            agent.max_speed = self.max_speed
        # add landmarks (goals)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
        # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle {i}"
            obstacle.collide = True
            obstacle.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)

        #################### set colours ####################
        # set colours for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set colours for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        # set colours for obstacles
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        #####################################################

        ####### set random positions for entities ###########
        # set random obstacles first
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)
        # set agents at random positions not colliding with obstacles
        num_agents_added = 0
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(-1, +1, world.dim_p)
            agent_size = world.agents[num_agents_added].size
            if not self.is_obstacle_collision(random_pos, agent_size, world):
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                num_agents_added += 1
        # set landmarks at random positions not colliding with obstacles and
        # also check collisions with placed goals
        num_goals_added = 0
        goals_added = []
        while True:
            if num_goals_added == self.num_agents:
                break
            random_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            goal_size = world.landmarks[num_goals_added].size
            obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
            landmark_collision = self.is_landmark_collision(
                random_pos, goal_size, world.landmarks[:num_goals_added]
            )
            if not landmark_collision and not obs_collision:
                world.landmarks[num_goals_added].state.p_pos = random_pos
                world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
                num_goals_added += 1
        #####################################################

        ############ find minimum times to goals ############
        if self.max_speed is not None:
            for agent in world.agents:
                self.min_time(agent, world)
        #####################################################

    def info_callback(self, agent: Agent, world: World) -> Tuple:
        # TODO modify this
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        goal = world.get_entity("landmark", agent.id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - goal.state.p_pos)))
        world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
            world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            "Dist_to_goal": world.dist_left_to_goal[agent.id],
            "Time_req_to_goal": world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obst_collisions": world.num_obstacle_collisions[agent.id],
        }
        if self.max_speed is not None:
            agent_info["Min_time_to_goal"] = agent.goal_min_time
        return agent_info

    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of agent with another agent
    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_landmark_collision(self, pos, size: float, landmark_list: List) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + landmark.size
            if dist < dist_min:
                collision = True
                break
        return collision

    # get min time required to reach to goal without obstacles
    def min_time(self, agent: Agent, world: World) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed"
        agent_id = agent.id
        # get the goal associated to this agent
        landmark = world.get_entity(entity_type="landmark", id=agent_id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
        min_time = dist / agent.max_speed
        agent.goal_min_time = min_time
        return min_time

    # done condition for each agent
    def done(self, agent: Agent, world: World) -> bool:
        # if we are using dones then return appropriate done
        if self.use_dones:
            landmark = world.get_entity("landmark", agent.id)
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
            if dist < self.min_dist_thresh:
                return True
            else:
                return False
        # it not using dones then return done
        # only when episode_length is reached
        else:
            if world.current_time_step >= world.world_length:
                return True
            else:
                return False

    def reward(self, agent: Agent, world: World) -> float:
        # Agents are rewarded based on distance to
        # its landmark, penalized for collisions
        rew = 0
        agents_goal = world.get_entity(entity_type="landmark", id=agent.id)
        dist_to_goal = np.sqrt(
            np.sum(np.square(agent.state.p_pos - agents_goal.state.p_pos))
        )
        if dist_to_goal < self.min_dist_thresh:
            rew += self.goal_rew
        else:
            rew -= dist_to_goal
        if agent.collide:
            for a in world.agents:
                # do not consider collision with itself
                if a.id == agent.id:
                    continue
                if self.is_collision(a, agent):
                    rew -= self.collision_rew

            if self.is_obstacle_collision(
                pos=agent.state.p_pos, entity_size=agent.size, world=world
            ):
                rew -= self.collision_rew
        return rew

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """
        agent: Agent object
        world: World object
        if obs_type=='local':
            return: agent_vel, agent_pos, agent_goal_pos
        if obs_type=='global':
            include information about all entities in the world:
            agent_vel, agent_pos, goal_pos, [other_agents_pos], [obstacles_pos]
        if obs_type=='nbd':
            return: agent_vel, agent_pos, agent_goal_pos, [other_entities_in_nbd_pos]
        """
        # get positions of all entities in this agent's reference frame
        agent_vel, agent_pos = agent.state.p_vel, agent.state.p_pos
        agents_goal = world.get_entity("landmark", agent.id)
        goal_pos = agents_goal.state.p_pos - agent_pos
        # goal_pos.append(agents_goal.state.p_pos - agent_pos)
        # communication of all other agents
        comm, other_agents_pos, obstacle_pos = [], [], []
        dist_mag = []
        if self.obs_type == "local":
            other_pos = (
                other_agents_pos + obstacle_pos
            )  # remember this is just concatenation of lists
        else:
            # get position of all other agents in this agent's reference frame
            for other in world.agents:
                if other is agent:
                    continue
                comm.append(other.state.c)
                other_agents_pos.append(np.array(other.state.p_pos - agent_pos))
                dist_mag.append(np.linalg.norm(other_agents_pos[-1]))
            # get position of all obstacles in this agent's reference frame
            obstacle_pos = []
            for obstacle in world.obstacles:
                obstacle_pos.append(np.array(obstacle.state.p_pos - agent_pos))
                dist_mag.append(np.linalg.norm(obstacle_pos[-1]))
            other_pos = np.array(
                other_agents_pos + obstacle_pos
            )  # remember + is just concatenation of lists
            if self.obs_type == "nbd":
                # sort other_pos according to distance from agent
                dist_mag_sort, dist_sort_idx = np.sort(dist_mag), np.argsort(dist_mag)
                other_pos = other_pos[dist_sort_idx, :]
                filter = np.array(dist_mag_sort) < self.max_edge_dist
                filter = np.expand_dims(filter, axis=1)  # shape (num_entities, 1)
                filter = np.repeat(filter, axis=1, repeats=other_pos.shape[1])
                # all pos_entities outside max_edge_dist are set to 0
                other_pos = other_pos * filter
                # only take the closes num_nbd_entities
                other_pos = other_pos[: self.num_nbd_entities, :]
            other_pos = other_pos.flatten()
        if self.use_comm:
            return np.concatenate(
                [agent_vel, agent_pos, goal_pos, other_pos, np.array(comm).flatten()]
            )
        else:
            return np.concatenate([agent_vel, agent_pos, goal_pos, other_pos])
        # return np.concatenate([agent.state.p_vel, agent.state.p_pos] + goal_pos + other_pos + comm)

    def shared_observation(self, world: World) -> np.ndarray:
        """
        Get master state of the environment
        [agents_pos], [obstacle_pos], [agents_goals]
        NOTE: Not including agent velocites because it is already present
        in agent's local information
        """
        # get agent positions
        agents_pos = []
        agents_goal = []
        obstacle_pos = []
        for agent in world.agents:
            agents_pos.append(agent.state.p_pos)
            agents_goal.append(world.get_entity("landmark", agent.id).state.p_pos)
        for obstacle in world.obstacles:
            obstacle_pos.append(obstacle.state.p_pos)

        return np.concatenate(agents_pos + agents_goal + obstacle_pos)


if __name__ == "__main__":
    from multiagent.environment import MultiAgentCADRLEnv
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
            self.share_env = False
            self.obs_type = "local"
            self.use_comm = False
            self.num_nbd_entities = 3
            self.max_edge_dist = 1
            self.px: int = 10
            self.py: int = 10
            self.vx: int = 1
            self.vy: int = 1
            self.radius: int = 1
            self.pgx: int = 10
            self.pgy: int = 10
            self.v_pref: int = 1
            self.theta: int = 0
            self.kinematic: bool = True

    args = Args()

    scenario = Scenario()

    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentCADRLEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback,
        done_callback=scenario.done,
        shared_viewer=False,
    )
    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    if args.share_env:
        obs_n, shared_obs = env.reset()
    else:
        obs_n = env.reset()
    stp = 0
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        if args.share_env:
            obs_n, shared_obs, reward_n, done_n, info_n = env.step(act_n)
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)
        # render all agent views
        env.render()
        stp += 1
