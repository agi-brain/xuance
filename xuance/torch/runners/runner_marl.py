"""
This is demo of runner for cooperative multi-agent reinforcement learning.
"""
import os
import socket
from pathlib import Path
from .runner_basic import Runner_Base
from xuance.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from copy import deepcopy


class Runner_MARL(Runner_Base):
    def __init__(self, args):
        super(Runner_MARL, self).__init__(args)
        self.args = args
        self.render = args.render
        self.test_envs = None

        time_string = time.asctime().replace(" ", "").replace(":", "_")
        seed = f"seed_{self.args.seed}_"
        self.args.model_dir_load = args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), args.model_dir, seed + time_string)
        if (not os.path.exists(self.args.model_dir_save)) and (not args.test_mode):
            os.makedirs(self.args.model_dir_save)

        if args.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), args.log_dir, seed + time_string)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.log_dir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=args.seed,
                       reinit=True)
            self.use_wandb = True
        else:
            raise RuntimeError(f"The logger named {args.logger} is implemented!")

        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.env_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_length
        self.num_agents = self.envs.num_agents
        args.n_agents = self.num_agents
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape, args.act_shape = (self.dim_obs,), ()
        args.rew_shape = args.done_shape = (1,)
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space

        # Create MARL agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)
        self.on_policy = self.agents.on_policy


