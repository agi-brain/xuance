import os
from pathlib import Path
from abc import ABC, abstractmethod
from argparse import Namespace
from xuance.common import Optional, Union
from xuance.mindspore import ms, nn, Tensor, Module, optim, ops


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training

        self.episode_length = config.episode_length
        self.learning_rate = getattr(config, "learning_rate", 0.001)
        self.use_linear_lr_decay = getattr(config, "use_linear_lr_decay", False)
        self.end_factor_lr_decay = getattr(config, "end_factor_lr_decay", 1.0)
        self.gamma = getattr(config, "gamma", 0.99)
        self.use_cnn = getattr(config, "use_cnn", False)
        self.use_rnn = getattr(config, "use_rnn", False)
        self.use_actions_mask = getattr(config, "use_actions_mask", False)
        self.model = model
        self.optimizer: Union[dict, list, Optional[nn.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[optim.lr_scheduler.LRScheduler]] = None
        self.callback = callback

        if self.distributed_training:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = self.device = int(os.environ['RANK'])
            self.snapshot_path = os.path.join(os.getcwd(), config.model_dir, "DDP_Snapshot")
            if os.path.exists(self.snapshot_path):
                if os.path.exists(os.path.join(self.snapshot_path, "snapshot.pt")):
                    print("Loading Snapshot...")
                    self.load_snapshot(self.snapshot_path)
            else:
                if self.device == 0:
                    os.makedirs(self.snapshot_path)
        else:
            self.world_size = 1
            self.rank = 0
            self.device = config.device

        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.total_iters = self.estimate_total_iterations()
        self.iterations = 0

    def get_grad_reducer(self,
                         optimizer: Union[dict, list, Optional[nn.Optimizer]]
                         ) -> Optional[nn.DistributedGradReducer]:
        if self.distributed_training:
            mean = ms.context.get_auto_parallel_context("gradients_mean")
            return nn.DistributedGradReducer(optimizer.parameters, mean)
        else:
            return None

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        start_training = getattr(self.config, "start_training", 0)
        training_frequency = getattr(self.config, "training_frequency", 1)
        total_iters = (self.config.running_steps - start_training) // (training_frequency * self.config.parallels)
        return total_iters

    def save_model(self, model_path):
        model_path += ".ckpt"
        ms.save_checkpoint(self.model, model_path)

    def load_model(self, path, model=None):
        path = os.path.join(path, model) if model is not None else path
        path += ".ckpt"

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
        ms.load_param_into_net(self.policy, ms.load_checkpoint(model_path))
        print(f"Successfully load model from '{path}'.")
        return path

    def load_snapshot(self, snapshot_path):
        pass

    def save_snapshot(self):
        pass

    def _safe_scheduler_step(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
