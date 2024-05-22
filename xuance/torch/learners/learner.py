import torch
import time
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from argparse import Namespace
import os


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
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./"):
        self.value_normalizer = None
        self.args = config
        self.n_agents = config.n_agents
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = torch.device("cuda" if (torch.cuda.is_available() and self.args.device == "gpu") else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(config.gpu_id)
            print("Use cuda, gpu ID: ", config.gpu_id)

        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

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
