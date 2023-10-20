import socket
import time
from pathlib import Path
from xuance.tensorflow.agents import *


class Agent(ABC):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Gym,
                 policy: tk.Model,
                 memory: Buffer,
                 learner: Learner,
                 device: str = "cpu",
                 logdir: str = "./logs/",
                 modeldir: str = "./models/"):
        self.config = config
        self.envs = envs
        self.policy = policy
        self.memory = memory
        self.learner = learner

        self.observation_space = envs.observation_space
        self.comm = MPI.COMM_WORLD
        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range
        self.returns = np.zeros((self.envs.num_envs,), np.float32)

        # logger
        if config.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), config.logdir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
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
                       group=config.env_id,
                       job_type=config.agent,
                       name=time.asctime(),
                       reinit=True
                       )
            # os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.device = device
        self.logdir = logdir
        self.modeldir = modeldir
        create_directory(logdir)
        create_directory(modeldir)
        self.current_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
    
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

    def log_videos(self, info: dict, fps: int, x_index: int=0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, Dict):
                for key in self.observation_space.spaces.keys():
                    observations[key] = np.clip(
                        (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                        -self.obsnorm_range, self.obsnorm_range)
            else:
                observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                       -self.obsnorm_range, self.obsnorm_range)
            return observations
        else:
            return observations

    def _process_reward(self, rewards):
        if self.use_rewnorm:
            std = np.clip(self.ret_rms.std, 0.1, 100)
            return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
        else:
            return rewards

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
