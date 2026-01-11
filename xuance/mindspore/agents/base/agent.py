import os.path
import wandb
import socket
import xuance
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from argparse import Namespace
from gymnasium.spaces import Dict, Space
from torch.utils.tensorboard import SummaryWriter
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.mindspore import REGISTRY_Representation, REGISTRY_Learners, Module, ms
from xuance.mindspore.utils import InitializeFunctions, NormalizeFunctions, ActivationFunctions, set_seed


class Agent(ABC):
    """Base class for single-agent Deep Reinforcement Learning (DRL).

    This class defines the common interface and shared infrastructure for
    single-agent DRL algorithms in XuanCe. An Agent encapsulates the policy,
    learner, and training/testing logic, while environments are managed
    externally by the runner or provided explicitly by the user.

    The agent can be initialized either with training environments (`envs`)
    or, for inference/testing-only scenarios, without environments but with
    explicit observation and action spaces.

    Args:
        config (Namespace): Configuration object containing hyperparameters,
            runtime settings, and environment specifications.
        envs (Optional[DummyVecEnv | SubprocVecEnv]): Vectorized environments
            used for training. If None, the agent will not initialize training
            environments and must be provided with `observation_space` and
            `action_space` to build networks.
        observation_space (Optional[gymnasium.spaces.Space]): Observation space
            specification used to construct policy networks when `envs` is None.
            Typically obtained from `test_envs.observation_space`.
        action_space (Optional[gymnasium.spaces.Space]): Action space
            specification used to construct policy networks when `envs` is None.
            Typically obtained from `test_envs.action_space`.
        callback (Optional[BaseCallback]): Optional callback object for injecting
            custom logic during training or evaluation (e.g., logging, early
            stopping, or custom hooks).

    Notes:
        - When `envs` is provided, the agent assumes a training context and
          derives observation/action spaces from the environments.
        - When `envs` is None, the agent can still be used for evaluation or
          inference as long as the corresponding spaces are explicitly given.
        - Environment creation and lifecycle management are intentionally
          decoupled from the agent and handled by the runner or user code.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        set_seed(config.seed)
        self.meta_data = dict(algo=config.agent, env=config.env_name, env_id=config.env_id,
                              dl_toolbox=config.dl_toolbox, device=config.device, seed=config.seed,
                              xuance_version=xuance.__version__)
        # Training settings.
        self.config = config
        self.use_rnn = config.use_rnn if hasattr(config, "use_rnn") else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, "use_actions_mask") else False
        self.distributed_training = getattr(config, "distributed_training", False)

        self.gamma = config.gamma
        self.start_training = config.start_training if hasattr(config, "start_training") else 1
        self.training_frequency = config.training_frequency if hasattr(config, "start_training") else 1
        self.n_epochs = config.n_epochs if hasattr(config, "n_epochs") else 1
        self.static_graph = getattr(config, "static_graph", True)
        self.device = config.device

        # Environment attributes.
        self.train_envs = envs
        self.render = config.render
        self.fps = config.fps
        if self.train_envs is None:
            if observation_space is None or action_space is None:
                raise ValueError("Please provide the observation_space and action_space when the envs is not provided."
                                 "Or the networks cannot be built."
                                 "You can get them from test_envs.observation_space and test_envs.action_space.")
            self.n_envs = self.config.parallels
            self.observation_space = observation_space
            self.action_space = action_space
            self.episode_length = self.config.episode_length = None
        else:
            self.train_envs.reset()
            self.n_envs = self.train_envs.num_envs
            self.episode_length = self.config.episode_length = self.train_envs.max_episode_steps
            self.observation_space = self.train_envs.observation_space
            self.action_space = self.train_envs.action_space
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

        # Set normalizations for observations and rewards.
        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space))
        self.ret_rms = RunningMeanStd(shape=())
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range
        self.returns = np.zeros((self.train_envs.num_envs,), np.float32)

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
        if self.static_graph:
            ms.set_context(mode=ms.GRAPH_MODE)  # Static graph mode (accelerating the calculation)
            print("Running mode: Static Graph. (Also known as Graph mode)")
        else:
            ms.set_context(mode=ms.PYNATIVE_MODE)  # Dynamic graph mode (default mode)
            print("Running mode: Dynamic Graph.")
        self.policy: Optional[Module] = None
        self.learner: Optional[Module] = None
        self.memory: Optional[object] = None
        self.callback = callback or BaseCallback()

    def save_model(self, model_name, model_path=None):
        # save the neural networks
        model_path = self.model_dir_save if model_path is None else model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.learner.save_model(os.path.join(model_path, model_name))
        # save the observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(model_path, "obs_rms.npy")
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

    @abstractmethod
    def _build_policy(self) -> Module:
        raise NotImplementedError

    def _build_learner(self, *args):
        return REGISTRY_Learners[self.config.learner](*args)

    @abstractmethod
    def action(self, observations):
        raise NotImplementedError

    @abstractmethod
    def train(self, train_steps: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def test(self,
             test_episodes: int,
             test_envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
             close_envs: bool = True):
        raise NotImplementedError

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()
        self.train_envs.close()
