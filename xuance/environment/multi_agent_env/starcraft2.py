import numpy as np
from smac.env import StarCraft2Env
from xuance.environment import RawMultiAgentEnv
from gym.spaces import Box, Discrete


class StarCraft2_Env(RawMultiAgentEnv):
    """
    The implementation of StarCraft2 environments, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
    """
    def __init__(self, config):
        super(StarCraft2_Env, self).__init__()
        self.env = StarCraft2Env(map_name=config.env_id)
        self.env_info = self.env.get_env_info()

        self.num_agents = self.env_info['n_agents']
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(self.env_info['state_shape'], ))
        self.observation_space = {k: Box(low=-np.inf, high=np.inf, shape=(self.env_info['obs_shape'], ))
                                  for k in self.agents}
        self.action_space = {k: Discrete(n=self.env_info['n_actions']) for k in self.agents}
        self.teams_info = {
            "names": ['agent'],
            "num_teams": 1,
            "agents_in_team": self.agents
        }
        self.max_episode_steps = self.env_info['episode_limit']

        self.n_agents = self.env_info["n_agents"]
        self.num_enemies = self.env.n_enemies

        self._episode_step = 0
        self.buf_info = {
            'battle_won': 0,
            'dead_allies': 0,
            'dead_enemies': 0,
        }

    def reset(self):
        """ Resets the environment. """
        obs, _ = self.env.reset()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}
        self._episode_step = 0
        info = {}
        return obs_dict, info

    def step(self, actions):
        """ Takes actions as input, perform a step in the underlying StarCraft2 environment. """
        actions_list = [actions[key] for key in self.agents]
        reward, terminated, info = self.env.step(actions_list)
        reward_dict = {k: reward for k in self.agents}
        terminated_dict = {k: terminated for k in self.agents}
        obs = self.env.get_obs()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}

        step_info = info
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False
        return obs_dict, reward_dict, terminated_dict, truncated, step_info

    def render(self, mode):
        """
        Renders the environment.

        Return:
            rgb_images (np.ndarray or list): The images used to visualize the environment.
        """
        return self.env.render(mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def state(self):
        """Returns the global state of the environment."""
        return self.env.get_state()

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        actions_mask_list = self.env.get_avail_actions()
        return {key: actions_mask_list[index] for index, key in enumerate(self.agents)}

