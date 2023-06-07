import gym


class Gym_Env(gym.Wrapper):
    def __init__(self, env_id: str, seed: int, render_mode: str):
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env.action_space.seed(seed=seed)
        self.env.reset(seed=seed)
        super(Gym_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range

    def close(self):
        self.env.close()
