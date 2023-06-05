from xuanpolicy.torch.agents import *


class Agent(ABC):
    def __init__(self,
                 envs: VecEnv,
                 policy: nn.Module,
                 memory: Buffer,
                 learner: Learner,
                 writer: SummaryWriter,
                 device: Optional[Union[str, int, torch.device]] = None,
                 logdir: str = "./logs/",
                 modeldir: str = "./models/",
                 ):
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

    def log_tb(self, infos: dict, n_steps: int):
        """
        infos: (dict) information to be visualized
        n_steps: current step
        """
        for k, v in infos.items():
            self.writer.add_scalars(k, {k: v}, n_steps)

    def log_wandb(self, infos: dict, n_steps: int):
        """
        infos: (dict) information to be visualized
        n_steps: current step
        """
        for k, v in infos.items():
            wandb.log({k, v}, step=n_steps)

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
    if agent_name in ["A2C", "A3C", "PG", "PPO_Clip", "PPO_KL", "PPG", "VDAC", "COMA", "MFAC", "MAPPO_Clip",
                      "MAPPO_KL"]:
        return int(args.training_steps * args.nepoch * args.nminibatch / args.nsteps)
    else:
        return int(args.training_steps / args.training_frequency)
