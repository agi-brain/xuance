from xuance.environment import RawMultiAgentEnv
from pettingzoo.utils.env import ParallelEnv
import numpy as np
import ctypes
import importlib

TEAM_NAME_DICT = {
    "mpe.simple_adversary_v3": ['adversary', 'agent'],
    "mpe.simple_crypto_v3": ['eve', 'alice', 'bob'],
    "mpe.simple_push_v3": ['adversary', 'agent'],
    "mpe.simple_reference_v3": ['agent'],
    "mpe.simple_speaker_listener_v4": ['speaker', 'listener'],
    "mpe.simple_spread_v3": ['agent'],
    "mpe.simple_tag_v3": ['adversary', 'agent'],
    "mpe.simple_v3": ['agent'],
    "mpe.simple_world_comm_v3": ['adversary', 'agent'],
}


class MPE_Env(RawMultiAgentEnv):
    """
    A wrapper for PettingZoo environments, provide a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning
    Parameters:
        env_name (str) – the name of the PettingZoo environment.
        env_id (str) – environment id.
        seed (int) – use to control randomness within the environment.
        kwargs (dict) – a variable-length keyword argument.
    """
    def __init__(self, config):
        super(MPE_Env, self).__init__()
        # Prepare raw environment
        env_name, env_id = config.env_name, config.env_id
        self.render_mode = config.render_mode
        self.continuous_actions = config.continuous_action
        self.scenario_name = env_name + "." + env_id
        scenario = importlib.import_module(f'pettingzoo.{env_name}.{env_id}')  # create scenario
        self.env = scenario.parallel_env(continuous_actions=self.continuous_actions, render_mode=self.render_mode)
        self.env.reset(config.seed)

        # Set basic attributes
        self.metadata = self.env.metadata
        self.state_space = self.env.state_space
        self.observation_space = self.env.observation_spaces
        self.action_space = self.env.action_spaces
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.team_info = {
            "names": TEAM_NAME_DICT[self.scenario_name],
            "num_teams": len(TEAM_NAME_DICT[self.scenario_name]),
            "agents_in_team": self.get_agents_in_team
        }
        self.max_episode_steps = self.env.unwrapped.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self):
        """Reset the environment to its initial state."""
        observations, infos = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {"infos": infos,
                      "individual_episode_rewards": self.individual_episode_reward}
        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        if self.continuous_actions:
            for k, v in actions.items():
                actions[k] = np.clip(v, self.action_spaces[k].low, self.action_spaces[k].high)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"infos": infos,
                     "individual_episode_rewards": self.individual_episode_reward}
        return observations, rewards, terminations, truncations, step_info

    def state(self):
        """Returns the global state of the environment."""
        return self.env.state()

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        Note: For MPE environment, all agents are alive before the episode is terminated.
        """
        mask = np.ones(self.num_agents, dtype=np.bool_)
        return mask

    def avail_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        if self.continuous_actions:
            return None
        else:
            avail_actions = {}
            for agent in self.agents:
                avail_actions[agent] = np.ones(self.action_space[agent].n, np.bool_)
            return avail_actions
