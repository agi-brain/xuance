from xuanpolicy.mindspore.agents import *


class Agent(ABC):
    def __init__(self,
                 envs: VecEnv,
                 policy: nn.Cell,
                 memory: Buffer,
                 learner: Learner,
                 writer: SummaryWriter,
                 logdir: str = "./logs/",
                 modeldir: str = "./models/",
                 ):
        self.envs = envs
        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.writer = writer
        self.logdir = logdir
        self.modeldir = modeldir
        create_directory(logdir)
        create_directory(modeldir)

    def save_model(self):
        self.learner.save_model()

    def load_model(self, path):
        self.learner.load_model(path)

    @abstractmethod
    def _process_observation(self, observations):
        raise NotImplementedError

    @abstractmethod
    def _process_reward(self, rewards):
        raise NotImplementedError

    @abstractmethod
    def _action(self, observations):
        raise NotImplementedError

    @abstractmethod
    def train(self, steps):
        raise NotImplementedError

    @abstractmethod
    def test(self, env, episodes):
        raise NotImplementedError
