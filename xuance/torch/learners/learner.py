import os
import torch
import torch.nn.functional as F
from numpy import concatenate as concat
from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Union
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor


class Learner(ABC):
    def __init__(self,
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
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
        self.policy.load_state_dict(torch.load(model_path, map_location={
            "cuda:0": self.device,
            "cuda:1": self.device,
            "cuda:2": self.device,
            "cuda:3": self.device,
            "cuda:4": self.device,
            "cuda:5": self.device,
            "cuda:6": self.device,
            "cuda:7": self.device
        }))
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
                 policy: torch.nn.Module,
                 optimizer: Optional[dict],
                 scheduler: Optional[dict] = None):
        self.value_normalizer = None
        self.config = config
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.use_parameter_sharing = config.use_parameter_sharing
        self.model_keys = model_keys
        self.agent_keys = agent_keys
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
                            use_actions_mask: Optional[bool] = False):
        """
        Prepare the training data.
        """
        batch_size = sample['batch_size']
        if use_parameter_sharing:
            k = self.model_keys[0]
            getter = itemgetter(*self.agent_keys)
            obs = {k: Tensor(concat([getter(sample['obs'])[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            actions = {k: Tensor(concat([getter(sample['actions'])[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            obs_next = {k: Tensor(concat([getter(sample['obs_next'])[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            rewards = {k: Tensor(concat([getter(sample['rewards'])[i][:, None, None] for i in range(self.n_agents)], 1)).to(self.device)}
            terminals = {k: Tensor(concat([getter(sample['terminals'])[i][:, None, None] for i in range(self.n_agents)], 1)).to(self.device)}
            agent_mask = {k: Tensor(concat([getter(sample['agent_mask'])[i][:, None, None] for i in range(self.n_agents)], 1)).to(self.device)}
            if use_actions_mask:
                avail_actions = {
                    k: Tensor(concat([getter(sample['avail_actions'])[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
                avail_actions_next = {
                    k: Tensor(concat([getter(sample['avail_actions_next'])[i][:, None] for i in range(self.n_agents)], 1)).to(self.device)}
            else:
                avail_actions, avail_actions_next = None, None
            IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
            rewards = {k: Tensor(sample['rewards'][k][:, None]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k][:, None]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k][:, None]).float().to(self.device) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}
                avail_actions_next = {k: Tensor(sample['avail_actions_next'][k]).float().to(self.device)
                                      for k in self.model_keys}
            else:
                avail_actions, avail_actions_next = None, None
            IDs = None

        sample_Tensor = {
            'batch_size': batch_size,
            'obs': obs,
            'actions': actions,
            'obs_next': obs_next,
            'rewards': rewards,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'avail_actions_next': avail_actions_next,
            'agent_ids': IDs,
        }
        return sample_Tensor

    def onehot_action(self, actions_int, num_actions):
        return F.one_hot(actions_int.long(), num_classes=num_actions)

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
        self.policy.load_state_dict(torch.load(model_path, map_location={
            "cuda:0": self.device,
            "cuda:1": self.device,
            "cuda:2": self.device,
            "cuda:3": self.device,
            "cuda:4": self.device,
            "cuda:5": self.device,
            "cuda:6": self.device,
            "cuda:7": self.device
        }))
        print(f"Successfully load model from '{path}'.")

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def update_recurrent(self, *args):
        pass

    def act(self, *args, **kwargs):
        pass

    def get_hidden_states(self, *args):
        pass

    def lr_decay(self, *args):
        pass
