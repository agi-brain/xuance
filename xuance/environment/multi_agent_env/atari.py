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
        from pettingzoo.atari import basketball_pong_v3
        self.env = basketball_pong_v3.parallel_env(num_players=2, render_mode="rgb_array")
        self.env = supersuit.max_observation_v0(self.env, 2)
        self.env = supersuit.sticky_actions_v0(self.env, repeat_action_probability=0.25)
        self.env = supersuit.frame_skip_v0(self.env, 4)
        self.env = supersuit.resize_v1(self.env, 84, 84)
        self.env = supersuit.frame_stack_v1(self.env, 4)
        self.env.reset()

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self, **kwargs):
        """Reset the environment to its initial state."""
        return self.env.reset(**kwargs)

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        return self.env.step(actions)

    def state(self):
        """Returns the global state of the environment."""
        return

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        """
        return

    def available_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        return





