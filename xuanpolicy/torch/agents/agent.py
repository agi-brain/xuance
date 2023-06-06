import socket
from xuanpolicy.torch.agents import *

class Agent(ABC):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Module,
                 memory: Buffer,
                 learner: Learner,
                 device: Optional[Union[str, int, torch.device]] = None,
                 logdir: str = "./logs/",
                 modeldir: str = "./models/",
                 ):
        self.envs = envs
        self.policy = policy
        self.memory = memory
        self.learner = learner
        if config.logger == "tensorboard":
            self.writer = SummaryWriter(config.logdir)
            self.use_wandb = False
        elif config.logger == "wandb":
            config_dict = vars(config)
            wandb.init(config=config_dict,
                       project=config.project_name,
                       entity=config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=os.path.join(os.getcwd(), config.logdir),
                       group=config.env_name,
                       job_type="training",
                       name=config.env_id,
                       reinit=True
                       )
            self.use_wandb = True
        else:
            raise "No logger is implemented."
        self.device = device
        self.logdir = logdir
        self.modeldir = modeldir
        create_directory(logdir)
        create_directory(modeldir)

    def save_model(self):
        self.learner.save_model()

    def load_model(self, path):
        self.learner.load_model(path)

    def log_infos(self, infos: dict, x_index: int):
        """
        infos: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in infos.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

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
