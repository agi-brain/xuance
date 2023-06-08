import socket
from pathlib import Path
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
            wandb_dir = Path(os.path.join(os.getcwd(), config.logdir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=config.project_name,
                       entity=config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=config.env,
                       job_type=config.agent,
                       name="seed_" + str(config.seed),
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

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def log_videos(self, info: dict, fps: int):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')})
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps)

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
    def test(self, steps):
        raise NotImplementedError


def get_total_iters(agent_name, args):
    if agent_name in ["A2C", "A3C", "PG", "PPO_Clip", "PPO_KL", "PPG", "VDAC", "COMA", "MFAC", "MAPPO_Clip",
                      "MAPPO_KL"]:
        return int(args.training_steps * args.nepoch * args.nminibatch / args.nsteps)
    else:
        return int(args.training_steps / args.training_frequency)
