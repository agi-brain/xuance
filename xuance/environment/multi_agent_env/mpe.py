import importlib
import numpy as np
from xuance.environment import RawMultiAgentEnv


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
    The implementation of MPE environments, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
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
        self.agents = self.env.agents
        self.state_space = self.env.state_space
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_space = {agent: self.env.action_space(agent) for agent in self.agents}
        self.num_agents = self.env.num_agents
        self.team_info = {
            "names": TEAM_NAME_DICT[self.scenario_name],
            "num_teams": len(TEAM_NAME_DICT[self.scenario_name]),
            "agents_in_team": self.get_agents_in_team
        }
        self.max_episode_steps = self.env.unwrapped.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_step = 0

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self):
        """Reset the environment to its initial state."""
        observations, infos = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {"infos": infos,
                      "individual_episode_rewards": self.individual_episode_reward}
        self._episode_step = 0
        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        if self.continuous_actions:
            for k, v in actions.items():
                actions[k] = np.clip(v, self.action_space[k].low, self.action_space[k].high)
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"infos": info,
                     "individual_episode_rewards": self.individual_episode_reward}
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False
        return observations, rewards, terminated, truncated, step_info

    def state(self):
        """Returns the global state of the environment."""
        return self.env.state()

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        Note: For MPE environment, all agents are alive before the episode is terminated.
        """
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        if self.continuous_actions:
            return None
        else:
            return {agent: np.ones(self.action_space[agent].n, np.bool_) for agent in self.agents}
