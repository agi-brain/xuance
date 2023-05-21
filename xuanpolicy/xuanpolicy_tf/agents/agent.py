from xuanpolicy.xuanpolicy_tf.agents import *

class Agent(ABC):
    def __init__(self,
                 envs: VecEnv,
                 policy: tk.Model,
                 memory: Buffer,
                 learner: Learner,
                 writer: tf.summary.SummaryWriter,
                 device: str = "cpu",
                 logdir: str = "./logs/",
                 modeldir: str = "./models/"):
        self.envs = envs
        self.policy = policy
        self.memory = memory
        self.learner = learner
        self.writer = writer
        self.device = device
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


def get_total_iters(agent_name, args):
    if agent_name in ["A2C", "PG", "PPO_Clip","PPO_KL","PPG","VDAC", "COMA", "MFAC", "MAPPO_Clip", "MAPPO_KL"]:
        return int(args.training_steps * args.nepoch * args.nminibatch / args.nsteps)
    else:
        return int(args.training_steps / args.training_frequency)
