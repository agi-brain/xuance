import os
import socket
from pathlib import Path
from .runner_basic import Runner_Base
from xuanpolicy.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np


class SC2_Runner(Runner_Base):
    def __init__(self, args):
        super(SC2_Runner, self).__init__(args)

        if args.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), args.logdir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.logdir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=time.asctime(),
                       reinit=True
                       )
            # os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.current_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)

        # environment details, representations, policies, optimizers, and agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)

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

    def train_episode(self, n_episodes):
        pass

    def test_episodes(self, n_episodes):
        pass

    def run(self):
        pass

    def benchmark(self):
        pass

