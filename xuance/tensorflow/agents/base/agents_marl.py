import os.path
import wandb
import socket
import xuance
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from argparse import Namespace
from typing import Optional, List, Dict, Union, Tuple
from gymnasium.spaces import Space
from xuance.common import get_time_string, create_directory, MultiAgentBaseCallback, AgentGrouping
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, space2shape
import tensorflow as tf
from xuance.tensorflow import Module, REGISTRY_Representation, REGISTRY_Learners
from xuance.tensorflow.learners import LearnerMAS
from xuance.tensorflow.utils import (normalizerFunctions, ActivationFunctions, initializerFunctions, AgentGroupedTensor,
                                     set_seed, set_device)
from xuance.tensorflow.rl_models import AgentFeatureEncoder
from xuance.tensorflow.rl_models import IdentityFeatureFusion, build_identity_encoder


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
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
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

        # TensorFlow distributed execution is normally controlled through
        # tf.distribute.Strategy at runner level.
        self.distributed_training = getattr(config, "distributed_training", False)
        self.strategy = getattr(config, "strategy", None)

        if self.distributed_training:
            if self.strategy is None:
                self.strategy = tf.distribute.get_strategy()
            self.world_size = int(self.strategy.num_replicas_in_sync)
            self.rank = int(os.environ.get("RANK", 0))
        else:
            self.strategy = tf.distribute.get_strategy()
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

        # Agent grouping.
        self.agent_grouping = self.set_agent_group(
            self.agent_keys
        )
        self.groups = self.agent_grouping.groups
        self.group_keys = self.agent_grouping.group_keys
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}

        with tf.device(self.device):
            self.agent_indices = {
                k: tf.convert_to_tensor(self.agent_grouping.agent_indices(k), dtype=tf.int64)
                for k in self.group_keys
            }

        # Network helpers.
        self.normalizer_fn = normalizerFunctions[self.config.normalizer] if hasattr(self.config, "normalizer") else None
        self.initializer = initializerFunctions[getattr(self.config, "initializer", "orthogonal")]
        self.activation = ActivationFunctions[self.config.activation]

        # Prepare directories.
        # A common time string can be passed by the runner for
        # multi-worker training through config.run_time_string.
        time_string = get_time_string()
        seed = f"seed_{config.seed}_"
        self.model_dir_load = config.model_dir
        self.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)

        # Create logger.
        if config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)

            if self.rank == 0:
                create_directory(log_dir)

            self.writer = tf.summary.create_file_writer(log_dir)
            self.use_wandb = False

        elif config.logger == "wandb":
            config_dict = vars(config)
            log_dir = config.log_dir
            wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))

            if self.rank == 0:
                create_directory(str(wandb_dir))
                wandb.init(
                    config=config_dict,
                    project=config.project_name,
                    entity=config.wandb_user_name,
                    notes=socket.gethostname(),
                    dir=wandb_dir,
                    group=config.env_id,
                    job_type=config.agent,
                    name=time_string,
                    reinit=True,
                )
            self.use_wandb = True

        else:
            raise AttributeError("No logger is implemented.")
        self.log_dir = log_dir

        # Predefine necessary components.
        self.model: Optional[Module] = None
        self.learner: Optional[LearnerMAS] = None
        self.memory: Optional[object] = None
        self.callback = callback or MultiAgentBaseCallback()

        self.meta_data = dict(
            algo=self.config.agent,
            env=self.config.env_name,
            env_id=self.config.env_id,
            dl_toolbox=self.config.dl_toolbox,
            device=self.device,
            seed=self.config.seed,
            xuance_version=xuance.__version__,
        )

    def set_agent_group(self, agent_keys):
        if self.use_parameter_sharing:
            return AgentGrouping.shared(agent_keys)
        return AgentGrouping.independent(agent_keys)

    @abstractmethod
    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name, model_path=None):
        if self.distributed_training and self.rank > 0:
            return

        model_path = self.model_dir_save if model_path is None else model_path

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.learner.save_model(os.path.join(model_path, model_name))

    def load_model(self, path, model=None):
        self.learner.load_model(path, model)

    def log_infos(self, info: dict, x_index: int):
        """Log scalar or grouped information."""
        if self.use_wandb:
            if self.rank != 0:
                return
            for key, value in info.items():
                if value is None:
                    continue
                if isinstance(value, (tf.Tensor, tf.Variable)):
                    value = value.numpy()
                wandb.log({key: value}, step=x_index)
        else:
            with self.writer.as_default():
                for key, value in info.items():
                    if value is None:
                        continue
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (tf.Tensor, tf.Variable)):
                                sub_value = sub_value.numpy()
                            tf.summary.scalar(f"{key}/{sub_key}", sub_value, step=x_index)
                    else:
                        if isinstance(value, (tf.Tensor, tf.Variable)):
                            value = value.numpy()
                        tf.summary.scalar(key, value, step=x_index)
                self.writer.flush()

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            if self.rank != 0:
                return
            for key, value in info.items():
                if value is None:
                    continue
                wandb.log({key: wandb.Video(value, fps=fps, format="gif")}, step=x_index)
        else:
            # TensorFlow summary has no direct equivalent of
            # SummaryWriter.add_video(). Log representative frames.
            with self.writer.as_default():
                for key, value in info.items():
                    if value is None:
                        continue
                    value = tf.convert_to_tensor(value)
                    # Common RL video shape:
                    # [N, T, C, H, W].
                    if value.shape.rank == 5:
                        frame = value[:, 0]
                        if frame.shape.rank == 4 and frame.shape[1] in (1, 3, 4):
                            frame = tf.transpose(frame, [0, 2, 3, 1])
                        tf.summary.image(key, frame, step=x_index, max_outputs=4)
                self.writer.flush()

    def _build_representation(self,
                              representation_choice: str,
                              input_space: Union[Dict[str, Space], Dict[str, tuple]],
                              config: Namespace) -> Module:
        """
        Build representation for policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            config: The configurations for creating the representation module.

        Returns:
            representation (Module): The representation Module.
        """

        # build representations
        input_representations = dict(
            input_shape=space2shape(input_space),
            hidden_sizes=getattr(config, "representation_hidden_size", None),
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation,
            kernels=getattr(config, "kernels", None),
            strides=getattr(config, "strides", None),
            filters=getattr(config, "filters", None),
            fc_hidden_sizes=getattr(config, "fc_hidden_sizes", None),
            N_recurrent_layers=getattr(config, "N_recurrent_layers", None),
            recurrent_hidden_size=getattr(config, "recurrent_hidden_size", None),
            rnn=getattr(config, "rnn", None),
            dropout=getattr(config, "dropout", None)
        )
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
                      avail_actions_list: Optional[List[dict]] = None
                      ):
        """Build inputs for representations before calculating actions.

        Args:
            obs_list: Observations of all vectorized environments.
            avail_actions_list: Available-action masks.

        Returns:
            Tuple containing grouped observations, agent indices,
            and grouped available-action masks.
        """
        batch_size = len(obs_list)
        obs_input = {}
        agent_indices = {}
        avail_actions = {} if self.use_actions_mask else None

        with tf.device(self.device):
            for group, group_agents in self.groups.items():
                obs_array = np.array([[obs[k] for k in group_agents] for obs in obs_list])
                obs_input[group] = tf.convert_to_tensor(obs_array)

                # [n_agents] -> [batch_size, n_agents, 1]
                indices = tf.tile(self.agent_indices[group][None, :], [batch_size, 1])
                indices = tf.reshape(indices, [batch_size, -1, 1])
                agent_indices[group] = indices

                if self.use_rnn:
                    # sequence length T = 1
                    obs_input[group] = tf.expand_dims(obs_input[group], axis=2)
                    agent_indices[group] = tf.expand_dims(agent_indices[group], axis=2)

                if self.use_actions_mask:
                    avail_array = np.array([[avail_a[k] for k in group_agents] for avail_a in avail_actions_list])
                    avail_actions[group] = tf.convert_to_tensor(avail_array)

                    if self.use_rnn:
                        avail_actions[group] = tf.expand_dims(avail_actions[group], axis=2)

        if self.use_actions_mask:
            grouped_avail_actions = AgentGroupedTensor(avail_actions, self.agent_grouping)
        else:
            grouped_avail_actions = None

        return (AgentGroupedTensor(obs_input, self.agent_grouping),
                AgentGroupedTensor(agent_indices, self.agent_grouping),
                grouped_avail_actions)

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
            if self.rank == 0:
                wandb.finish()
        else:
            self.writer.flush()
            self.writer.close()


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
