import os
import torch
from numpy import concatenate as concat
from abc import ABC, abstractmethod
from typing import Optional, List, Union
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor

MAX_GPUs = 100


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 episode_length: int,
                 policy: torch.nn.Module,
                 optimizer: Union[dict, Optional[torch.optim.Optimizer]],
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        self.value_normalizer = None
        self.config = config

        self.episode_length = episode_length
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, model=None):
        file_names = os.listdir(path)
        if model is not None:
            path = os.path.join(path, model)
            if model not in file_names:
                raise RuntimeError(f"The folder '{path}' does not exist, please specify a correct path to load model.")
        else:
            for f in file_names:
                if "seed_" not in f:
                    file_names.remove(f)
            file_names.sort()
            path = os.path.join(path, file_names[-1])

        model_names = os.listdir(path)
        if os.path.exists(path + "/obs_rms.npy"):
            model_names.remove("obs_rms.npy")
        if len(model_names) == 0:
            raise RuntimeError(f"There is no model file in '{path}'!")
        model_names.sort()
        model_path = os.path.join(path, model_names[-1])
        self.policy.load_state_dict(torch.load(str(model_path), map_location={
            f"cuda:{i}": self.device for i in range(MAX_GPUs)}))
        print(f"Successfully load model from '{path}'.")
        return path

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 episode_length: int,
                 policy: torch.nn.Module,
                 optimizer: Union[dict, Optional[torch.optim.Optimizer]],
                 scheduler: Union[dict, Optional[torch.optim.lr_scheduler.LinearLR]] = None):
        self.value_normalizer = None
        self.config = config
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.use_parameter_sharing = config.use_parameter_sharing
        self.model_keys = model_keys
        self.agent_keys = agent_keys
        self.episode_length = episode_length
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_parameter_sharing (bool): Whether to use parameter sharing for individual agent models.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            sample_Tensor (dict): The formatted sampled data.
        """
        batch_size = sample['batch_size']
        state, avail_actions, filled, seq_length = None, None, None, 0
        obs_next, state_next, avail_actions_next = None, None, None
        IDs = None
        if use_parameter_sharing:
            k = self.model_keys[0]
            obs_array = itemgetter(*self.agent_keys)(sample['obs'])
            obs = {k: Tensor(concat([obs_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            act_array = itemgetter(*self.agent_keys)(sample['actions'])
            actions = {k: Tensor(concat([act_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            rew_array = itemgetter(*self.agent_keys)(sample['rewards'])
            ter_array = itemgetter(*self.agent_keys)(sample['terminals'])
            agt_mask_array = itemgetter(*self.agent_keys)(sample['agent_mask'])
            if self.use_rnn:
                seq_length = act_array[0].shape[1]
                rewards = {k: Tensor(concat([rew_array[i][:, None, :, None] for i in range(self.n_agents)], 1)).to(self.device)}
                terminals = {k: Tensor(concat([ter_array[i][:, None, :, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
                agent_mask = {k: Tensor(concat([agt_mask_array[i][:, None, :, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, seq_length + 1, -1).to(self.device)
            else:
                obs_next_array = itemgetter(*self.agent_keys)(sample['obs_next'])
                obs_next = {k: Tensor(concat([obs_next_array[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
                rewards = {k: Tensor(concat([rew_array[i][:, None, None] for i in range(self.n_agents)], 1)).to(self.device)}
                terminals = {k: Tensor(concat([ter_array[i][:, None, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
                agent_mask = {k: Tensor(concat([agt_mask_array[i][:, None, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

            if use_actions_mask:
                act_mask_array = itemgetter(*self.agent_keys)(sample['avail_actions'])
                avail_actions = {k: Tensor(concat([act_mask_array[i][:, None] for i in range(self.n_agents)], 1)).float().to(self.device)}
                if not self.use_rnn:
                    act_mask_next_array = itemgetter(*self.agent_keys)(sample['avail_actions'])
                    avail_actions_next = {k: Tensor(concat([act_mask_next_array[i][:, None] for i in range(self.n_agents)], 1)).float().to(self.device)}

        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            if self.use_rnn:
                rewards = {k: Tensor(sample['rewards'][k][:, :, None]).to(self.device) for k in self.agent_keys}
                terminals = {k: Tensor(sample['terminals'][k][:, :, None]).float().to(self.device) for k in self.agent_keys}
                agent_mask = {k: Tensor(sample['agent_mask'][k][:, :, None]).float().to(self.device) for k in self.agent_keys}
            else:
                obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
                rewards = {k: Tensor(sample['rewards'][k][:, None]).to(self.device) for k in self.agent_keys}
                terminals = {k: Tensor(sample['terminals'][k][:, None]).float().to(self.device) for k in self.agent_keys}
                agent_mask = {k: Tensor(sample['agent_mask'][k][:, None]).float().to(self.device) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}
                if not self.use_rnn:
                    avail_actions_next = {k: Tensor(sample['avail_actions_next'][k]).float().to(self.device) for k in self.model_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)
            if not self.use_rnn:
                state_next = Tensor(sample['state_next']).to(self.device)

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'state_next': state_next,
            'obs': obs,
            'actions': actions,
            'obs_next': obs_next,
            'rewards': rewards,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'avail_actions_next': avail_actions_next,
            'agent_ids': IDs,
            'filled': filled,
        }
        return sample_Tensor

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def update_rnn(self, *args):
        raise NotImplementedError

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, model=None):
        file_names = os.listdir(path)
        if model is not None:
            path = os.path.join(path, model)
            if model not in file_names:
                raise RuntimeError(f"The folder '{path}' does not exist, please specify a correct path to load model.")
        else:
            for f in file_names:
                if "seed_" not in f:
                    file_names.remove(f)
            file_names.sort()
            path = os.path.join(path, file_names[-1])

        model_names = os.listdir(path)
        if os.path.exists(path + "/obs_rms.npy"):
            model_names.remove("obs_rms.npy")
        if len(model_names) == 0:
            raise RuntimeError(f"There is no model file in '{path}'!")
        model_names.sort()
        model_path = os.path.join(path, model_names[-1])
        self.policy.load_state_dict(torch.load(str(model_path), map_location={
            f"cuda:{i}": self.device for i in range(MAX_GPUs)}))
        print(f"Successfully load model from '{path}'.")
