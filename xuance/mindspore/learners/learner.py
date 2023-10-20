import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import OneHot, Eye
import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
import os


class Learner(ABC):
    def __init__(self,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.modeldir = modeldir
        self.iterations = 0

    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ", "")
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        if not os.path.exists(self.modeldir):
            try:
                os.mkdir(self.modeldir)
            except:
                os.makedirs(self.modeldir)
        ms.save_checkpoint(self.policy, model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        model_names.remove('obs_rms.npy')
        model_names.sort()
        model_path = path + model_names[-1]
        ms.load_param_into_net(self.policy, ms.load_checkpoint(model_path))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Union[nn.Optimizer, Sequence[nn.Optimizer]],
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./"):
        self.args = config
        self.handle = config.handle
        self.n_agents = config.n_agents
        self.agent_keys = config.agent_keys
        self.agent_index = config.agent_ids
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents

        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.modeldir = modeldir
        self.iterations = 0
        self._one_hot = OneHot()
        self.eye = Eye()
        self.expand_dims = ms.ops.ExpandDims()

    def onehot_action(self, actions_int, num_actions):
        return self._one_hot(actions_int.astype(ms.int32), num_actions,
                             ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32))

    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ", "")
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        if not os.path.exists(self.modeldir):
            try:
                os.mkdir(self.modeldir)
            except:
                os.makedirs(self.modeldir)
        ms.save_checkpoint(self.policy, model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        # model_names.remove('obs_rms.npy')
        model_names.sort()
        model_path = path + model_names[-1]
        ms.load_param_into_net(self.policy, ms.load_checkpoint(model_path))

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError