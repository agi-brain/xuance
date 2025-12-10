from xuance.environment import RawMultiAgentEnv


class AtariMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(AtariMultiAgentEnv, self).__init__()
        from pettingzoo.atari import basketball_pong_v3
        self.env = basketball_pong_v3.env(num_players=2, render_mode="rgb_array")
        self.env.reset()
        self.env.reset(config.env_seed)

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, mode='human'):
        return self.env.render()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)



