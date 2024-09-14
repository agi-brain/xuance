import os.path
import wandb
import socket
import numpy as np
from abc import ABC
from pathlib import Path
from argparse import Namespace
from mpi4py import MPI
from gym.spaces import Dict, Space
from torch.utils.tensorboard import SummaryWriter
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional
from xuance.environment import DummyVecEnv
from xuance.mindspore import REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.mindspore.utils import InitializeFunctions, NormalizeFunctions, ActivationFunctions


class Agent(ABC):
    """Base class of agent for single-agent DRL.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        # Training settings.
        self.config = config
        self.use_rnn = config.use_rnn if hasattr(config, "use_rnn") else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, "use_actions_mask") else False

        self.gamma = config.gamma
        self.start_training = config.start_training if hasattr(config, "start_training") else 1
        self.training_frequency = config.training_frequency if hasattr(config, "start_training") else 1
        self.n_epochs = config.n_epochs if hasattr(config, "n_epochs") else 1
        self.device = config.device

        # Environment attributes.
        self.envs = envs
        self.envs.reset()
        self.episode_length = self.config.episode_length = envs.max_episode_steps
        self.render = config.render
        self.fps = config.fps
        self.n_envs = envs.num_envs
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

        # Set normalizations for observations and rewards.
        self.comm = MPI.COMM_WORLD
        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range
        self.returns = np.zeros((self.envs.num_envs,), np.float32)

        # Prepare directories.
        time_string = get_time_string()
        seed = f"seed_{self.config.seed}_"
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

        # Prepare necessary components.
        self.policy: Optional[Module] = None
        self.learner: Optional[Module] = None
        self.memory: Optional[object] = None

    def save_model(self, model_name):
        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)
        # save the observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(self.model_dir_save, "obs_rms.npy")
            observation_stat = {'count': self.obs_rms.count,
                                'mean': self.obs_rms.mean,
                                'var': self.obs_rms.var}
            np.save(obs_norm_path, observation_stat)

    def load_model(self, path, model=None):
        path_loaded = self.learner.load_model(path, model)
        # recover observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(path_loaded, "obs_rms.npy")
            if os.path.exists(obs_norm_path):
                observation_stat = np.load(obs_norm_path, allow_pickle=True).item()
                self.obs_rms.count = observation_stat['count']
                self.obs_rms.mean = observation_stat['mean']
                self.obs_rms.var = observation_stat['var']
            else:
                raise RuntimeError(f"Failed to load observation status file 'obs_rms.npy' from {obs_norm_path}!")

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                if v is None:
                    continue
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

    def _build_representation(self, representation_key: str,
                              input_space: Optional[Space],
                              config: Namespace):
        """
        Build representation for policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            input_space (Optional[Space]): The space of input tensors.
            config: The configurations for creating the representation module.

        Returns:
            representation (Module): The representation Module.
        """
        input_representations = dict(
            input_shape=space2shape(input_space),
            hidden_sizes=config.representation_hidden_size if hasattr(config, "representation_hidden_size") else None,
            normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
            initialize=InitializeFunctions['orthogonal'] if hasattr(config, "orthogonal") else None,
            activation=ActivationFunctions[config.activation],
            kernels=config.kernels if hasattr(config, "kernels") else None,
            strides=config.strides if hasattr(config, "strides") else None,
            filters=config.filters if hasattr(config, "filters") else None,
            fc_hidden_sizes=config.fc_hidden_sizes if hasattr(config, "fc_hidden_sizes") else None,
            device=self.device)
        representation = REGISTRY_Representation[representation_key](**input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def _build_learner(self, *args) -> Module:
        return REGISTRY_Learners[self.config.learner](*args)

    def action(self, observations):
        raise NotImplementedError

    def train(self, steps):
        raise NotImplementedError

    def test(self, env_fn, steps):
        raise NotImplementedError

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()
        self.envs.close()
