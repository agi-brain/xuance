import os.path
import wandb
import socket
import numpy as np
from abc import ABC
from pathlib import Path
from argparse import Namespace
from typing import Optional
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from xuance.common import get_time_string, create_directory
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch import ModuleDict


class MARLAgents(ABC):
    """Base class of agents for MARL.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        # Training settings.
        self.config = config
        self.use_rnn = config.use_rnn if hasattr(config, "use_rnn") else False
        self.use_parameter_sharing = config.use_parameter_sharing
        self.use_actions_mask = config.use_actions_mask if hasattr(config, "use_actions_mask") else False

        self.gamma = config.gamma
        self.start_training = config.start_training if hasattr(config, "start_training") else 1
        self.training_frequency = config.training_frequency if hasattr(config, "training_frequency") else 1
        self.device = config.device

        # Environment attributes.
        self.envs = envs
        self.envs.reset()
        self.n_agents = config.n_agents
        self.render = config.render
        self.fps = config.fps
        self.n_envs = envs.num_envs
        self.agent_keys = envs.agents
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.episode_length = config.episode_length if hasattr(config, "episode_length") else envs.max_episode_steps
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

        # Prepare directories.
        time_string = get_time_string()
        seed = f"seed_{config.seed}_"
        self.model_dir_load = config.model_dir
        self.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)

        # Create logger.
        if config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
            create_directory(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif config.logger == "wandb":
            config_dict = vars(config)
            log_dir = config.log_dir
            wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
            create_directory(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=config.project_name,
                       entity=config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=config.env_id,
                       job_type=config.agent,
                       name=time_string,
                       reinit=True,
                       settings=wandb.Settings(start_method="fork")
                       )
            # os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise AttributeError("No logger is implemented.")
        self.log_dir = log_dir

        # predefine necessary components
        self.model_keys = [self.agent_keys[0]] if self.use_parameter_sharing else self.agent_keys
        self.policy: Optional[nn.Module] = None
        self.learner: Optional[nn.Module] = None
        self.memory: Optional[object] = None

    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name):
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)

    def load_model(self, path, model=None):
        self.learner.load_model(path, model)

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

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)
                
    def _build_representation(self, representation_key: str, config: Namespace):
        """
        Build representation for policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            config: The configurations for creating the representation module.
        
        Returns:
            representation (Module): The representation Module. 
        """
        normalize_fn = NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None
        initializer = nn.init.orthogonal_
        activation = ActivationFunctions[config.activation]
        device = self.device
        agent = config.agent

        # build representations
        representation = ModuleDict()
        for key in self.model_keys:
            input_shape = self.observation_space[key].shape
            if representation_key == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif representation_key == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif representation_key == "Basic_RNN":
                representation[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
            else:
                raise AttributeError(f"{agent} currently does not support {representation_key} representation.")
        return representation

    def _build_policy(self):
        raise NotImplementedError

    def _build_learner(self, *args):
        raise NotImplementedError

    def action(self, **kwargs):
        raise NotImplementedError

    def train_epochs(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def test(self, **kwargs):
        raise NotImplementedError

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()


class RandomAgents(object):
    def __init__(self, args, envs, device=None):
        self.args = args
        self.n_agents = self.args.n_agents
        self.agent_keys = args.agent_keys
        self.action_space = self.args.action_space
        self.nenvs = envs.num_envs

    def action(self, obs_n, episode, test_mode, noise=False):
        rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
        random_actions = np.array(rand_a)
        return random_actions

    def load_model(self, model_dir):
        return
