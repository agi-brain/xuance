import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from xuance.common import Optional, List, Union
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor

MAX_GPUs = 100


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: torch.nn.Module):
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training

        self.episode_length = config.episode_length
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer: Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None

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
        self.running_steps = config.running_steps
        self.iterations = 0

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)
        if self.distributed_training:
            self.save_snapshot()

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

    def load_snapshot(self, snapshot_path):
        loc = f"cuda: {self.device}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.policy.load_state_dict(snapshot["MODEL_STATE"])
        print("Resuming training from snapshot.")

    def save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.policy.state_dict(),
        }
        snapshot_pt = os.path.join(self.snapshot_path, "snapshot.pt")
        torch.save(snapshot, snapshot_pt)

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: torch.nn.Module):
        self.value_normalizer = None
        self.config = config
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.use_parameter_sharing = config.use_parameter_sharing
        self.model_keys = model_keys
        self.agent_keys = agent_keys
        self.episode_length = config.episode_length
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer: Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None
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
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled = None, None, None
        obs_next, state_next, avail_actions_next = None, None, None
        IDs = None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)).to(self.device)
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)).to(self.device)
            rewards_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['rewards']), axis=1)).to(self.device)
            ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1)).float().to(self.device)
            msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1)).float().to(self.device)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape(bs, seq_length + 1, -1)}
                if len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, seq_length)}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: actions_tensor.reshape(bs, seq_length, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents, seq_length)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents, seq_length)}
                agent_mask = {k: msk_tensor.reshape(bs, seq_length)}
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length + 1, -1).reshape(bs, seq_length + 1, self.n_agents).to(self.device)
            else:
                obs = {k: obs_tensor.reshape(bs, -1)}
                if len(actions_tensor.shape) == 2:
                    actions = {k: actions_tensor.reshape(bs)}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: rewards_tensor.reshape(batch_size, self.n_agents)}
                terminals = {k: ter_tensor.reshape(batch_size, self.n_agents)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                obs_next = {k: Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs_next']),
                                               axis=1)).to(self.device).reshape(bs, -1)}
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(
                    batch_size, -1, -1).reshape(bs, self.n_agents).to(self.device)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, seq_length + 1, -1])).float().to(self.device)}
                else:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, -1])).float().to(self.device)}
                    avail_a_next = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions_next']), axis=1)
                    avail_actions_next = {k: Tensor(avail_a_next.reshape([bs, -1])).float().to(self.device)}
        else:
            obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
            actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
            rewards = {k: Tensor(sample['rewards'][k]).to(self.device) for k in self.agent_keys}
            terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
            agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
            if not self.use_rnn:
                obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
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
            'seq_length': seq_length,
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
