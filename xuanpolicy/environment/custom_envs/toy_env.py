ENVIRONMENT_IDS = ['CartPole-v0', 'LunarLander-v2',
                   'Acrobot-v1', 'MountainCar-v0',
                   'Pendulum-v1', 'Platform-v0']
import gym


class Toy_Env(gym.Env):
    def __init__(self, env_id: str, seed: int):
        assert env_id in ENVIRONMENT_IDS
        if env_id == 'Platform-v0':
            import gym_platform
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        super(Toy_Env, self).__init__()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()
