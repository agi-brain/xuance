import os
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from xuance.common import Optional, Union
from argparse import Namespace
from xuance.torch import Module

MAX_GPUs = torch.cuda.device_count()


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training

        self.episode_length = config.episode_length
        self.learning_rate = config.learning_rate if hasattr(config, 'learning_rate') else None
        self.use_linear_lr_decay = config.use_linear_lr_decay if hasattr(config, 'use_linear_lr_decay') else False
        self.end_factor_lr_decay = config.end_factor_lr_decay if hasattr(config, 'end_factor_lr_decay') else 1.0
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.use_cnn = getattr(config, "use_cnn", False)
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.model = model
        self.optimizer: Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None
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

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        start_training = getattr(self.config, "start_training", 0)
        training_frequency = getattr(self.config, "training_frequency", 1)
        total_iters = (self.config.running_steps - start_training) // (training_frequency * self.config.parallels)
        return total_iters

    def save_model(self, model_path):
        if type(self.optimizer) is dict:
            torch.save(
                {
                    'policy': self.model.state_dict(),
                    'optimizer': {k: v.state_dict() for k, v in self.optimizer.items()},
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                },
                model_path)
        elif type(self.optimizer) is list:  # e.g. PDQN-family learners keep [actor, qnet] optimizers.
            torch.save(
                {
                    'policy': self.model.state_dict(),
                    'optimizer': [opt.state_dict() for opt in self.optimizer],
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                },
                model_path)
        else:
            torch.save(
                {
                    'policy': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                },
                model_path)
        if self.distributed_training:
            self.save_snapshot()

    def load_model(self, path, model=None):
        target_path = os.path.join(path, model) if model is not None else path
        if os.path.isfile(target_path):  # load the specified model file
            model_path = target_path
            dir_name = os.path.dirname(model_path)
        else:
            if not os.path.isdir(path):
                raise RuntimeError(f"The path '{path}' is not a valid directory or file!")
            folder_names = [f for f in os.listdir(path) if "seed_" in f]
            folder_names.sort()
            if not folder_names:
                raise RuntimeError(f"No model files with 'seed_' found in '{path}'!")
            path = Path(os.path.join(path, folder_names[-1]))
            dir_name = str(path)
            model_names = list(path.glob("*.pth"))
            model_path = None
            if len(model_names) == 0:
                raise FileNotFoundError(f"No .pth file found in {path}")
            else:
                for f in model_names:
                    if "final_train_model.pth" in str(f):
                        model_path = f
                        break
                    model_path = str(model_names)

        checkpoint = torch.load(str(model_path), map_location={f"cuda:{i}": self.device
                                                               for i in range(MAX_GPUs)}, weights_only=True)
        self.model.load_state_dict(checkpoint['policy'], strict=False)

        if 'optimizer' in checkpoint and self.optimizer is not None:
            if type(self.optimizer) is dict:
                for k, v in self.optimizer.items():
                    v.load_state_dict(checkpoint['optimizer'][k])
                current_lr = next(iter(self.optimizer.values())).param_groups[0]['lr']
            elif type(self.optimizer) is list:  # e.g. PDQN-family learners keep [actor, qnet] optimizers.
                for opt, opt_state in zip(self.optimizer, checkpoint['optimizer']):
                    opt.load_state_dict(opt_state)
                current_lr = self.optimizer[0].param_groups[0]['lr']
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate = current_lr

        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            rng_state = rng_state.cpu().to(dtype=torch.uint8)
            torch.set_rng_state(rng_state)

        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            cuda_states = checkpoint['cuda_rng_state']
            if isinstance(cuda_states, list):

                num_available_gpus = torch.cuda.device_count()
                cuda_states = cuda_states[:num_available_gpus]

                for i, state in enumerate(cuda_states):
                    state = state.cpu().to(dtype=torch.uint8)

                    torch.cuda.set_rng_state(state, device=i)

        self._safe_scheduler_step()
        print(f"Successfully load model from '{model_path}'.")
        return dir_name

    def load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        snapshot = torch.load(snapshot_path, map_location=loc)

        if "MODEL_STATE" in snapshot:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        elif "policy" in snapshot:
            self.model.load_state_dict(snapshot["policy"])

            if "optimizer" in snapshot and self.optimizer is not None:
                self.optimizer.load_state_dict(snapshot["optimizer"])

            if "rng_state" in snapshot:
                torch.set_rng_state(snapshot["rng_state"].to('cpu'))
            if "cuda_rng_state" in snapshot and torch.cuda.is_available():
                cuda_states = snapshot["cuda_rng_state"]
                if isinstance(cuda_states, list):
                    for i, state in enumerate(cuda_states):
                        torch.cuda.set_rng_state(state.to(f'cuda:{i}'), device=i)

        print("Resuming training from snapshot (including optimizer/rng state).")

    def save_snapshot(self):
        snapshot = {
            "policy": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        snapshot_pt = os.path.join(self.snapshot_path, "snapshot.pt")
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(snapshot, snapshot_pt)

    def _safe_scheduler_step(self):
        if not hasattr(self, 'scheduler'):
            return

        if not hasattr(self.config, 'rt_epoch'):
            return
        try:
            train_steps = self.config.running_steps // self.config.parallels
            eval_interval = self.config.eval_interval // self.config.parallels
            num_epoch = int(train_steps / eval_interval)
            current_iters = int(self.total_iters * self.config.rt_epoch / num_epoch)
            self.scheduler.step(current_iters)
            print(f"scheduler.step success，rt_epoch={self.config.rt_epoch}")
        except TypeError as e:
            if "positional argument" in str(e) or "takes 1 positional argument" in str(e):
                self.scheduler.step()
                print(f"scheduler.step success, rt_epoch={self.config.rt_epoch}")
        except Exception as e:
            print(f"scheduler.step failure：{e}")

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError
