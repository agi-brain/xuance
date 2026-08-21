import os.path
import wandb
import socket
import torch
import xuance
import numpy as np
import torch.distributed as dist
from abc import ABC, abstractmethod
from pathlib import Path
from argparse import Namespace
from typing import Optional, List, Dict, Union, Tuple
from gymnasium.spaces import Space
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import destroy_process_group
from xuance.common import get_time_string, create_directory, MultiAgentBaseCallback, AgentGrouping
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, space2shape
from xuance.torch import REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.torch.learners import LearnerMAS
from xuance.torch.utils import (NormalizeFunctions, InitializeFunctions, ActivationFunctions, AgentGroupedTensor,
                                init_distributed_mode, set_seed, set_device)
from xuance.torch.rl_models import AgentFeatureEncoder
from xuance.torch.rl_models import IdentityFeatureFusion, build_identity_encoder


class MARLAgents(ABC):
    """Base class for Multi-Agent Reinforcement Learning (MARL) agents.

    This class defines the common interface and shared functionalities for all
    MARL agent implementations in XuanCe. It handles environment interaction,
    logging, model saving/loading, distributed training setup, and representation
    construction, while leaving algorithm-specific logic to subclasses.

    Subclasses should implement the abstract methods to define:
        - how experiences are stored,
        - how actions are selected,
        - how training and evaluation are performed.

    Args:
        config (Namespace):
            A configuration object that contains hyperparameters and runtime
            settings, such as algorithm name, environment name, learning rates,
            device, seed, and logging options.
        envs (Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv]):
            Vectorized multi-agent environments for training. If not provided,
            environment-related attributes (e.g., observation/action spaces)
            must be specified explicitly.
        num_agents (Optional[int]):
            Number of agents in the environment. Required if `envs` is None.
        agent_keys (Optional[List[str]]):
            Unique identifiers for each agent. Required if `envs` is None.
        state_space (Optional[Space]):
            Global state space used by centralized critics or state-based
            representations. Required when `use_global_state` is enabled and
            `envs` is None.
        observation_space (Optional[Space]):
            Observation space for each agent. Required if `envs` is None.
        action_space (Optional[Space]):
            Action space for each agent. Required if `envs` is None.
        callback (Optional[MultiAgentBaseCallback]):
            A user-defined callback object for injecting custom logic during
            training and evaluation (e.g., logging, early stopping, debugging).
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: int = None,
            agent_keys: List[str] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        set_seed(config.seed)
        # Training settings.
        self.config = config
        self.use_cnn = getattr(config, "use_cnn", False)
        self.use_rnn = getattr(config, "use_rnn", False)
        self.use_parameter_sharing = config.use_parameter_sharing
        self.use_actions_mask = getattr(config, "use_actions_mask", False)
        self.use_global_state = getattr(config, "use_global_state", False)
        self.distributed_training = config.distributed_training
        if self.distributed_training:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            master_port = getattr(config, "master_port", None)
            init_distributed_mode(master_port=master_port)
        else:
            self.world_size = 1
            self.rank = 0

        self.gamma = config.gamma
        self.start_training = getattr(config, "start_training", 1)
        self.training_frequency = getattr(config, "training_frequency", 1)
        self.n_epochs = getattr(config, "n_epochs", 1)
        self.device = self.config.device = set_device(self.config.device)

        # Environment attributes.
        self.train_envs = envs
        self.render = config.render
        self.fps = config.fps
        if self.train_envs is None:
            if observation_space is None or action_space is None or agent_keys is None or num_agents is None:
                raise ValueError(
                    "Please provide the num_agents, agent_keys, observation_space, and action_space when the envs is not provided. Or the networks cannot be built."
                    "You can get them from test_envs.num_agents, test_envs.agents, test_envs.observation_space, and test_envs.action_space.")
            if self.use_global_state and state_space is None:
                raise ValueError("Please provide the state_space when the envs is not provided.")
            self.n_envs = self.config.parallels
            self.n_agents = self.config.n_agents = num_agents
            self.agent_keys = agent_keys
            self.state_space = state_space if self.use_global_state else None
            self.observation_space = observation_space
            self.action_space = action_space
            self.episode_length = None
        else:
            try:
                self.train_envs.reset()
            except:
                pass
            self.n_agents = self.config.n_agents = self.train_envs.num_agents
            self.n_envs = self.train_envs.num_envs
            self.agent_keys = self.train_envs.agents
            self.state_space = self.train_envs.state_space if self.use_global_state else None
            self.observation_space = self.train_envs.observation_space
            self.action_space = self.train_envs.action_space
            self.episode_length = getattr(config, "episode_length", self.train_envs.max_episode_steps)
        self.config.episode_length = self.episode_length
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

        self.agent_grouping = self.set_agent_group(self.agent_keys)
        self.groups = self.agent_grouping.groups
        self.group_keys = self.agent_grouping.group_keys
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.agent_indices = {k: torch.as_tensor(self.agent_grouping.agent_indices(k),
                                                 dtype=torch.int64, device=self.device)
                              for k in self.group_keys}

        # Set network's normalizer, initializer, activation.
        self.normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        self.initializer = InitializeFunctions[getattr(self.config, "initializer", "orthogonal")]
        self.activation = ActivationFunctions[self.config.activation]

        # Prepare directories.
        if self.distributed_training and self.world_size > 1:
            if self.rank == 0:
                time_string = get_time_string()
                time_string_tensor = torch.tensor(list(time_string.encode('utf-8')), dtype=torch.uint8).to(self.rank)
            else:
                time_string_tensor = torch.zeros(16, dtype=torch.uint8).to(self.rank)

            dist.broadcast(time_string_tensor, src=0)
            time_string = bytes(time_string_tensor.cpu().tolist()).decode('utf-8').rstrip('\x00')
        else:
            time_string = get_time_string()
        seed = f"seed_{config.seed}_"
        self.model_dir_load = config.model_dir
        self.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)

        # Create logger.
        if config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
            if self.rank == 0:
                create_directory(log_dir)
            else:
                while not os.path.exists(log_dir):
                    pass  # Wait until the master process finishes creating directory.
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif config.logger == "wandb":
            config_dict = vars(config)
            log_dir = config.log_dir
            wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
            if self.rank == 0:
                create_directory(str(wandb_dir))
            else:
                while not os.path.exists(str(wandb_dir)):
                    pass  # Wait until the master process finishes creating directory.
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
        self.model: Optional[nn.Module] = None
        self.learner: Optional[LearnerMAS] = None
        self.memory: Optional[object] = None
        self.callback = callback or MultiAgentBaseCallback()

        self.meta_data = dict(algo=self.config.agent, env=self.config.env_name, env_id=self.config.env_id,
                              dl_toolbox=self.config.dl_toolbox, device=self.device, seed=self.config.seed,
                              xuance_version=xuance.__version__)

    def set_agent_group(self, agent_keys):
        if self.use_parameter_sharing:
            agent_grouping = AgentGrouping.shared(agent_keys)
        else:
            agent_grouping = AgentGrouping.independent(agent_keys)
        return agent_grouping

    @abstractmethod
    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name, model_path=None):
        if self.distributed_training:
            if self.rank > 0:
                return

        # save the neural networks
        model_path = self.model_dir_save if model_path is None else model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.learner.save_model(os.path.join(model_path, model_name))

    def load_model(self, path, model=None):
        # load neural networks
        self.learner.load_model(path, model)

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

    def _build_representation(self,
                              representation_choice: str,
                              input_space: Union[Dict[str, Space], Dict[str, tuple]],
                              config: Namespace) -> Module:
        """
        Build representation for policies.

        Parameters:
            representation_choice (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            config: The configurations for creating the representation module.
        
        Returns:
            representation (Module): The representation Module. 
        """
        # build representations
        input_representations = dict(
            input_shape=space2shape(input_space),
            hidden_sizes=getattr(config, "representation_hidden_size", None),
            normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
            initialize=nn.init.orthogonal_,
            activation=ActivationFunctions[config.activation],
            kernels=getattr(config, "kernels", None),
            strides=getattr(config, "strides", None),
            filters=getattr(config, "filters", None),
            fc_hidden_sizes=getattr(config, "fc_hidden_sizes", None),
            N_recurrent_layers=getattr(config, "N_recurrent_layers", None),
            recurrent_hidden_size=getattr(config, "recurrent_hidden_size", None),
            rnn=getattr(config, "rnn", None),
            dropout=getattr(config, "dropout", None),
            device=self.device)
        representation = REGISTRY_Representation[representation_choice](**input_representations)
        if representation_choice not in REGISTRY_Representation:
            raise AttributeError(f"{representation_choice} is not registered in REGISTRY_Representation.")
        return representation

    def _build_agent_feature_encoder(
            self,
            representation_choice: str,
            group_agents: Tuple[str, ...],
            input_space: Union[Dict[str, Space], Dict[str, tuple], tuple]
    ) -> AgentFeatureEncoder:
        # build representations
        representation = self._build_representation(representation_choice,
                                                    input_space,
                                                    self.config)
        # build identity encoder
        agent_identity_encoder = build_identity_encoder(
            num_identities=len(group_agents),
            mode=getattr(self.config, "identity_embedding_mode", 'none'),
            embedding_dim=getattr(self.config, "identity_embedding_dim", None),
            device=self.device,
        )
        # build feature fusion
        identity_feature_fusion = IdentityFeatureFusion(
            observation_feature_dim=representation.output_shapes['state'][0],
            identity_feature_dim=agent_identity_encoder.output_dim,
            mode=getattr(self.config, "identity_feature_fusion_mode", "concat")
        )
        # build feature encoder
        return AgentFeatureEncoder(
            representation=representation,
            identity_encoder=agent_identity_encoder,
            fusion=identity_feature_fusion
        )

    @abstractmethod
    def _build_model(self) -> Module:
        raise NotImplementedError

    def _build_learner(self, *args):
        return REGISTRY_Learners[self.config.learner](*args)

    def _build_inputs(self,
                      obs_list: List[dict],
                      avail_actions_list: Optional[List[dict]] = None):
        """
        Build inputs for representations before calculating actions.

        Parameters:
            obs_list (List[dict]): List of observations for each agent in self.agent_keys.
            avail_actions_list (Optional[List[dict]]): Actions mask values, default is None.

        Returns:
            obs_input: The represented observations.
            agents_id: The agent id (One-Hot variables).
        """
        batch_size = len(obs_list)
        obs_input = {}
        agent_indices = {}
        avail_actions = {} if self.use_actions_mask else None

        for group, group_agents in self.groups.items():
            obs_input[group] = torch.as_tensor(np.array([[obs[k] for k in group_agents] for obs in obs_list]),
                                               device=self.device)  # shape: batch_size * n_agent * obs_dim

            agent_indices[group] = self.agent_indices[group].repeat(batch_size, 1).reshape(batch_size, -1, 1)
            if self.use_rnn:  # sequence length T=1
                obs_input[group] = obs_input[group].unsqueeze(2)  # shape: batch_size * n_agent * 1 * obs_dim
                agent_indices[group] = agent_indices[group].unsqueeze(2)  # shape: batch_size * n_agent * 1 * 1

            if self.use_actions_mask:
                avail_actions[group] = torch.as_tensor(np.array([[avail_a[k] for k in group_agents]
                                                                 for avail_a in avail_actions_list]),
                                                       device=self.device)  # shape: batch_size * n_agent * n_actions
                if self.use_rnn:
                    avail_actions[group] = avail_actions[group].unsqueeze(2)

        return (AgentGroupedTensor(obs_input, self.agent_grouping),
                AgentGroupedTensor(agent_indices, self.agent_grouping),
                AgentGroupedTensor(avail_actions, self.agent_grouping))

    @abstractmethod
    def get_actions(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_epochs(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def test(self, **kwargs):
        raise NotImplementedError

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()
        if self.distributed_training:
            if dist.get_rank() == 0:
                if os.path.exists(self.learner.snapshot_path):
                    if os.path.exists(os.path.join(self.learner.snapshot_path, "snapshot.pt")):
                        os.remove(os.path.join(self.learner.snapshot_path, "snapshot.pt"))
                    os.removedirs(self.learner.snapshot_path)
            destroy_process_group()


class RandomAgents(object):
    def __init__(self, args, envs, device=None):
        self.args = args
        self.n_agents = self.args.n_agents
        self.agent_keys = args.agent_keys
        self.action_space = self.args.action_space
        self.nenvs = envs.num_envs

    def get_actions(self, *args, **kwargs):
        rand_a = [[self.action_space[agent].sample() for agent in self.agent_keys] for e in range(self.nenvs)]
        random_actions = np.array(rand_a)
        return random_actions

    def load_model(self, model_dir):
        return
