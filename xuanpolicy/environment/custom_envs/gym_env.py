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


class Atari_Env(gym.Wrapper):
    def __init__(self,
                 env_id: str,
                 seed: int,
                 render_mode: str,
                 obs_type: str):
        self.env = gym.make(env_id,
                            render_mode=render_mode,
                            obs_type=obs_type,
                            frameskip=1)
        self.env.action_space.seed(seed=seed)
        self.env.reset(seed=seed)
        super(Atari_Env, self).__init__(self.env)
        # self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = render_mode

    def close(self):
        self.env.close()

    def reset(self):
        return self.env.reset(), {}

    def step(self, actions):
        observation, reward, terminated, info = self.env.unwrapped.step(actions)
        return observation, reward, terminated, terminated, info
