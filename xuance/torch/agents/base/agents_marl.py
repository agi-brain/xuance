import os.path
import wandb
import socket
import torch
import numpy as np
import torch.distributed as dist
from abc import ABC
from pathlib import Path
from argparse import Namespace
from operator import itemgetter
from gym.spaces import Space
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from xuance.common import get_time_string, create_directory, space2shape, Optional, List, Dict, Union
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import ModuleDict, REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.torch.learners import learner
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions, init_distributed_mode


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
        self.use_global_state = config.use_global_state if hasattr(config, "use_global_state") else False
        self.distributed_training = config.distributed_training
        if self.distributed_training:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            master_port = config.master_port if hasattr(config, "master_port") else None
            init_distributed_mode(master_port=master_port)
        else:
            self.world_size = 1
            self.rank = 0

        self.gamma = config.gamma
        self.start_training = config.start_training if hasattr(config, "start_training") else 1
        self.training_frequency = config.training_frequency if hasattr(config, "training_frequency") else 1
        self.n_epochs = config.n_epochs if hasattr(config, "n_epochs") else 1
        self.device = config.device

        # Environment attributes.
        self.envs = envs
        self.envs.reset()
        self.n_agents = self.config.n_agents = envs.num_agents
        self.render = config.render
        self.fps = config.fps
        self.n_envs = envs.num_envs
        self.agent_keys = envs.agents
        self.state_space = envs.state_space if self.use_global_state else None
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.episode_length = config.episode_length if hasattr(config, "episode_length") else envs.max_episode_steps
        self.config.episode_length = self.episode_length
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

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
        self.policy: Optional[nn.Module] = None
        self.learner: Optional[learner] = None
        self.memory: Optional[object] = None

    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name):
        if self.distributed_training:
            if self.rank > 0:
                return

        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)

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
                
    def _build_representation(self, representation_key: str,
                              input_space: Union[Dict[str, Space], tuple],
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
        representation = ModuleDict()
        for key in self.model_keys:
            if self.use_rnn:
                hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                'recurrent_hidden_size': self.config.recurrent_hidden_size}
            else:
                hidden_sizes = config.representation_hidden_size if hasattr(config,
                                                                            "representation_hidden_size") else None
            input_representations = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
                initialize=nn.init.orthogonal_,
                activation=ActivationFunctions[config.activation],
                kernels=config.kernels if hasattr(config, "kernels") else None,
                strides=config.strides if hasattr(config, "strides") else None,
                filters=config.filters if hasattr(config, "filters") else None,
                fc_hidden_sizes=config.fc_hidden_sizes if hasattr(config, "fc_hidden_sizes") else None,
                N_recurrent_layers=config.N_recurrent_layers if hasattr(config, "N_recurrent_layers") else None,
                rnn=config.rnn if hasattr(config, "rnn") else None,
                dropout=config.dropout if hasattr(config, "dropout") else None,
                device=self.device)
            representation[key] = REGISTRY_Representation[representation_key](**input_representations)
            if representation_key not in REGISTRY_Representation:
                raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def _build_learner(self, *args) -> Module:
        return REGISTRY_Learners[self.config.learner](*args)

    def _build_inputs(self,
                      obs_dict: List[dict],
                      avail_actions_dict: Optional[List[dict]] = None):
        """
        Build inputs for representations before calculating actions.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.

        Returns:
            obs_input: The represented observations.
            agents_id: The agent id (One-Hot variables).
        """
        batch_size = len(obs_dict)
        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        avail_actions_input = None

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            obs_array = np.array([itemgetter(*self.agent_keys)(data) for data in obs_dict])
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            avail_actions_array = np.array([itemgetter(*self.agent_keys)(data)
                                            for data in avail_actions_dict]) if self.use_actions_mask else None
            if self.use_rnn:
                obs_input = {key: obs_array.reshape([bs, 1, -1])}
                agents_id = agents_id.reshape(bs, 1, -1)
                if self.use_actions_mask:
                    avail_actions_input = {key: avail_actions_array.reshape([bs, 1, -1])}
            else:
                obs_input = {key: obs_array.reshape([bs, -1])}
                agents_id = agents_id.reshape(bs, -1)
                if self.use_actions_mask:
                    avail_actions_input = {key: avail_actions_array.reshape([bs, -1])}
        else:
            agents_id = None
            if self.use_rnn:
                obs_input = {k: np.stack([data[k] for data in obs_dict]).reshape([bs, 1, -1]) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.stack([data[k] for data in avail_actions_dict]).reshape([bs, 1, -1])
                                           for k in self.agent_keys}
            else:
                obs_input = {k: np.stack([data[k] for data in obs_dict]).reshape(bs, -1) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([data[k] for data in avail_actions_dict]).reshape([bs, -1])
                                           for k in self.agent_keys}
        return obs_input, agents_id, avail_actions_input

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
        self.envs.close()


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
