from xuance.mindspore.agents import *


class MARLAgents(object):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv,
                 policy: nn.Cell,
                 memory: BaseBuffer,
                 learner: LearnerMAS,
                 log_dir: str = "./logs/",
                 model_dir: str = "./models/"):
        self.args = config
        self.n_agents = config.n_agents
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.envs = envs
        self.start_training = config.start_training

        self.render = config.render
        self.nenvs = envs.num_envs
        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.log_dir = log_dir
        self.model_dir_save, self.model_dir_load = config.model_dir_save, config.model_dir_load
        create_directory(log_dir)
        create_directory(model_dir)

        self.eye = ms.ops.Eye()
        self.expand_dims = ms.ops.ExpandDims()

    def save_model(self, model_name):
        model_path = self.model_dir_save
        self.learner.save_model(model_path, model_name)

    def load_model(self, path, seed=1):
        self.learner.load_model(path, seed)

    def act(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class linear_decay_or_increase(object):
    def __init__(self, start, end, step_length):
        self.start = start
        self.end = end
        self.step_length = step_length
        if self.start > self.end:
            self.is_decay = True
            self.delta = (self.start - self.end) / self.step_length
        else:
            self.is_decay = False
            self.delta = (self.end - self.start) / self.step_length
        self.epsilon = start

    def update(self):
        if self.is_decay:
            self.epsilon = max(self.epsilon - self.delta, self.end)
        else:
            self.epsilon = min(self.epsilon + self.delta, self.end)


class RandomAgents(object):
    def __init__(self, args, envs):
        self.args = args
        self.n_agents = self.args.n_agents
        self.agent_keys = args.agent_keys
        self.action_space = self.args.action_space
        self.nenvs = envs.num_envs

    def act(self, obs_n, episode, test_mode, noise=False):
        rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
        random_actions = np.array(rand_a)
        return random_actions

    def load_model(self, model_dir):
        return
