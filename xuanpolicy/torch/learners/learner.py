import torch
import time
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
import os


class Learner(ABC):
    def __init__(self,
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0

    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ", "")
        model_path = self.modeldir + "model-%s-%s.pth" % (time.asctime(), str(self.iterations))
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        model_names.remove('obs_rms.npy')
        model_names.sort()
        model_path = path + model_names[-1]
        self.policy.load_state_dict(torch.load(model_path))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./"):
        self.args = config
        self.handle = config.handle
        self.n_agents = config.n_agents
        self.agent_keys = config.agent_keys
        self.agent_index = config.agent_ids
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
        self.writer = summary_writer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0

    def onehot_action(self, actions_int, num_actions):
        return F.one_hot(actions_int.long(), num_classes=num_actions)

    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ", "")
        model_path = self.modeldir + "model-%s-%s.pth" % (time.asctime(), str(self.iterations))
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        # model_names.remove('obs_rms.npy')
        model_names.sort()
        model_path = path + model_names[-1]
        self.policy.load_state_dict(torch.load(model_path))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError
