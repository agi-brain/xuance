from gfootball.env import _apply_output_wrappers
from .raw_env import football_raw_env
from gym.spaces import MultiDiscrete, Discrete
import numpy as np


class GFootball_Env:
    """The wrapper of original football environment.

    Args:
        args: the SimpleNamespace variable that contains attributes to create an original env.
    """
    def __init__(self, args):
        env = football_raw_env(args)
        self.env = _apply_output_wrappers(env=env,
                                          rewards=args.rewards_type,
                                          representation=args.obs_type,
                                          channel_dimensions=(args.smm_width, args.smm_height),
                                          apply_single_agent_wrappers=(args.num_agent + args.num_adversary == 1),
                                          stacked=args.num_adversary)
        self.n_agents = args.num_agent
        self.n_adversaries = args.num_adversary
        self.observation_space = self.env.observation_space
        self.dim_obs = self.observation_space.shape[-1]
        self.action_space = self.env.action_space
        if isinstance(self.action_space, MultiDiscrete):
            self.dim_act = self.n_actions = self.action_space.nvec[0]
        elif isinstance(self.action_space, Discrete):
            self.dim_act = self.n_actions = self.action_space.n
        else:
            raise "Unsupported action spaces"
        self.max_cycles = self.env.unwrapped.observation()[0]['steps_left']
        self._episode_step = 0
        self._episode_score = 0.0
        self.filled = np.zeros([self.max_cycles, 1], np.bool_)
        self.env.reset()
        state = self.get_state()
        self.dim_state = state.shape[0]
        self.dim_reward = self.n_agents

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args, **kwargs):
        """Get one-step frame."""
        return self.env.get_frame()

    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        obs = obs.reshape([self.n_agents, -1])
        state = self.get_state()
        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            "episode_step": self._episode_step,
            "episode_score": self._episode_score,
        }
        return obs, state, info

    def step(self, actions):
        """One-step transition of the environment.

        Args:
            actions: the actions for all agents.
        """
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = obs.reshape([self.n_agents, -1])
        state = self.get_state()
        self._episode_step += 1
        self._episode_score += reward.mean()
        info["episode_step"] = self._episode_step
        info["episode_score"] = self._episode_score
        truncated = True if self._episode_step >= self.max_cycles else False
        return obs, state, reward, terminated, truncated, info

    def get_more_info(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["active"] = np.array([state[i]['active'] for i in range(self.n_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.n_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.n_agents)])
        return info

    def get_state(self):
        """Get global state."""
        return np.array(self.env.env.state())

