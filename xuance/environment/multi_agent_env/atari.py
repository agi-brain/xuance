import numpy as np
import importlib
import supersuit
from xuance.environment import RawMultiAgentEnv


class AtariMultiAgentEnv(RawMultiAgentEnv):
    """
    The implementation of Atari environments with multiple players, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
    """

    def __init__(self, config):
        super(AtariMultiAgentEnv, self).__init__()
        # Prepare raw environment
        env_name, env_id = config.env_name, config.env_id
        self.render_mode = config.render_mode
        self.scenario_name = env_name + "." + env_id
        scenario = importlib.import_module(f'pettingzoo.{env_name}.{env_id}')  # create scenario
        self.obs_type = config.obs_type
        self.env = scenario.parallel_env(obs_type=self.obs_type, render_mode=self.render_mode)

        # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
        # to deal with frame flickering
        self.env = supersuit.max_observation_v0(self.env, 2)

        # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
        self.env = supersuit.sticky_actions_v0(self.env, repeat_action_probability=0.25)

        # skip frames for faster processing and less control
        # to be compatible with gym, use frame_skip(env, (2,5))
        self.frame_skip = config.frame_skip
        self.env = supersuit.frame_skip_v0(self.env, self.frame_skip)

        # downscale observation for faster processing
        self.image_size = getattr(config, 'img_size', [84, 84])
        assert config.img_size is not None
        self.env = supersuit.resize_v1(self.env, *self.image_size)
        self.env = supersuit.frame_stack_v1(self.env, self.frame_skip)
        self.env.reset()
        self._render_mode = config.render_mode

        self.metadata = self.env.metadata
        self.agents = self.env.agents
        self.state_space = self.env.observation_space(self.agents[0])
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_space = {agent: self.env.action_space(agent) for agent in self.agents}
        self.num_agents = self.env.num_agents
        self.max_episode_steps = self.env.unwrapped.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_score = 0.0

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self, **kwargs):
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
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        return self.env.step(actions)

    def state(self):
        """Returns the global state of the environment."""
        return

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        """
        return {agent: True for agent in self.agents}

    def available_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        return {agent: np.ones(self.action_space[agent].n, np.bool_) for agent in self.agents}





