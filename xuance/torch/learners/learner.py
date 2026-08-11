import os
from typing import Dict, Any

import torch
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from torch import Tensor

from xuance.common import Optional, Union, Dict, AgentGrouping
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor, Module

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
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.use_parameter_sharing = config.use_parameter_sharing
        self.agent_grouping = agent_grouping
        self.groups = agent_grouping.groups
        self.group_keys = agent_grouping.group_keys
        self.agent_keys = agent_grouping.agent_keys
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.agent_indices = {k: self.agent_grouping.agent_indices(k) for k in self.group_keys}

        self.episode_length = config.episode_length
        self.learning_rate = getattr(config, 'learning_rate', None)
        self.use_linear_lr_decay = getattr(config, 'use_linear_lr_decay', False)
        self.end_factor_lr_decay = getattr(config, 'end_factor_lr_decay', 0.5)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.use_cnn = getattr(config, "use_cnn", False)
        self.use_rnn = getattr(config, 'use_rnn', False)
        self.use_actions_mask = getattr(config, 'use_actions_mask', False)
        self.model = model
        self.optimizer: Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None
        self.callback = callback

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
        n_epochs = getattr(self.config, "n_epochs", 1)
        episode_length = self.episode_length
        if self.use_rnn:
            total_iters = (self.config.running_steps - start_training) // (episode_length * self.config.parallels)
        else:
            total_iters = (self.config.running_steps - start_training) // (training_frequency * self.config.parallels)
        total_iters *= n_epochs
        return total_iters

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
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, state_next, filled = None, None, None
        obs, obs_next, actions, rewards, terminals, agent_mask = {}, {}, {}, {}, {}, {}
        avail_actions = {} if self.use_actions_mask else None
        avail_actions_next = {} if use_actions_mask else None
        agent_indices = {}

        for agent in self.agent_keys:
            obs[agent] = torch.as_tensor(sample['obs'][agent], device=self.device)
            if not self.use_rnn:
                obs_next[agent] = torch.as_tensor(sample['obs_next'][agent], device=self.device)
            actions[agent] = torch.as_tensor(sample['actions'][agent], device=self.device)
            rewards[agent] = torch.as_tensor(sample['rewards'][agent], device=self.device)
            terminals[agent] = torch.as_tensor(sample['terminals'][agent], device=self.device, dtype=torch.float32)
            agent_mask[agent] = torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            if use_actions_mask:
                avail_actions[agent] = torch.as_tensor(sample['avail_actions'][agent],
                                                       device=self.device, dtype=torch.float32)
                if not self.use_rnn:
                    avail_actions_next[agent] = torch.as_tensor(sample['avail_actions_next'][agent],
                                                                device=self.device, dtype=torch.float32)

        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)
            if not self.use_rnn:
                state_next = torch.as_tensor(sample['state_next'], device=self.device)

        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        for key in self.group_keys:
            n_agents = self.n_group_agents[key]
            bs = batch_size * n_agents

            if self.use_rnn:
                agents_id = torch.as_tensor(self.agent_grouping.agent_indices(key), dtype=torch.int64).repeat(
                    batch_size, 1).reshape(bs, 1, 1).expand(-1, seq_length + 1, -1).to(self.device)
            else:
                agents_id = torch.as_tensor(self.agent_indices[key],
                                            dtype=torch.int64).repeat(batch_size, 1).reshape([bs, 1]).to(self.device)

            agent_indices[key] = agents_id

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
            'agent_indices': agent_indices,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def packed_tensor(self, agent_tensor: Dict[str, Tensor] = None) -> dict[str, Tensor] | None:
        if agent_tensor is None:
            return
        tensor_packed = {}
        tensor_shape = agent_tensor[self.agent_keys[0]].shape
        batch_size = tensor_shape[0]

        for key in self.group_keys:
            n_agents = self.n_group_agents[key]
            agent_keys = self.groups[key]
            bs = batch_size * n_agents

            tensor_group = torch.stack([agent_tensor[k] for k in agent_keys], dim=1)
            if self.use_cnn and len(tensor_group.shape) > 3:  # obs_array consists of images
                grouped_tensor_shape = (bs,) + tensor_shape[2:]
            else:
                grouped_tensor_shape = (bs,) + tensor_shape[1:]

            tensor_packed[key] = tensor_group.reshape(grouped_tensor_shape)

        return tensor_packed

    def get_joint_input(self, input_tensor, output_shape=None):
        if self.n_agents == 1:
            joint_tensor = itemgetter(*self.agent_keys)(input_tensor)
        else:
            joint_tensor = torch.concat(itemgetter(*self.agent_keys)(input_tensor), dim=-1)
        if output_shape is not None:
            joint_tensor = joint_tensor.reshape(output_shape)
        return joint_tensor

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def update_rnn(self, *args):
        raise NotImplementedError

    def save_model(self, model_path):
        if type(self.optimizer) is dict:
            if type(list(self.optimizer.values())[0]) is dict:
                torch.save(
                    {
                        'policy': self.model.state_dict(),
                        'optimizer': {k_a: {k: v.state_dict() for k, v in v_a.items()}
                                      for k_a, v_a in self.optimizer.items()},  # agent-wise
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    },
                    model_path)
            else:
                torch.save(
                    {
                        'policy': self.model.state_dict(),
                        'optimizer': {k: v.state_dict() for k, v in self.optimizer.items()},
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

        if self.device.upper() == "CPU":
            checkpoint = torch.load(str(model_path), map_location={'cuda:0': 'cpu'})
        else:
            checkpoint = torch.load(str(model_path), map_location={f"cuda:{i}": self.device
                                                                   for i in range(MAX_GPUs)}, weights_only=True)
        self.model.load_state_dict(checkpoint['policy'], strict=False)

        if 'optimizer' in checkpoint and self.optimizer is not None:
            if type(self.optimizer) is dict:
                if type(list(self.optimizer.values())[0]) is dict:
                    for k_a, v_a in self.optimizer.items():  # agent-wise
                        for k, v in v_a.items():
                            v.load_state_dict(checkpoint['optimizer'][k_a][k])
                    current_lr = list(self.optimizer.values())[0][
                        list(self.optimizer.values())[0].keys().__iter__().__next__()].param_groups[0]['lr']
                else:
                    for k, v in self.optimizer.items():
                        v.load_state_dict(checkpoint['optimizer'][k])
                    current_lr = list(self.optimizer.values())[0].param_groups[0]['lr']
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
