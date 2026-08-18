import os
import torch
import torch.nn as nn
from pathlib import Path
from abc import abstractmethod
from xuance.common import Optional, Dict, AgentGrouping
from argparse import Namespace
from operator import itemgetter
from xuance.torch import Tensor, Module
from xuance.torch.utils import ValueNorm
from xuance.torch.rl_models.modules import OnPolicyMARLBatch, OffPolicyMARLBatch
from xuance.torch.learners.base.drl_learner import Learner
from xuance.torch.utils import AgentGroupedTensor

MAX_GPUs = torch.cuda.device_count()


class LearnerMAS(Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(LearnerMAS, self).__init__(config, model, callback)
        self.use_parameter_sharing = config.use_parameter_sharing
        self.agent_grouping = agent_grouping
        self.groups = self.agent_grouping.groups
        self.group_keys = self.agent_grouping.group_keys
        self.agent_keys = self.agent_grouping.agent_keys
        self.n_agents = len(self.agent_keys)
        self.n_group_agents = {k: len(self.groups[k]) for k in self.group_keys}
        self.agent_indices = {k: torch.as_tensor(self.agent_grouping.agent_indices(k),
                                                 dtype=torch.int64, device=self.device)
                              for k in self.group_keys}

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

    def get_joint_input(self, input_tensor, output_shape=None):
        if self.n_agents == 1:
            joint_tensor = input_tensor[self.agent_keys[0]]
        else:
            joint_tensor = torch.concat([input_tensor[key] for key in self.agent_keys], dim=-1)
        if output_shape is not None:
            joint_tensor = joint_tensor.reshape(output_shape)
        return joint_tensor

    def build_training_data(self,
                            sample: Optional[dict],
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False) -> OnPolicyMARLBatch | OffPolicyMARLBatch:
        raise NotImplementedError

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=getattr(self.config, "weight_decay", 0.0)
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.end_factor_lr_decay,
            total_iters=self.total_iters
        )

    @abstractmethod
    def update(self, *args, **kwargs):
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


class OnPolicyMultiAgentLearner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(OnPolicyMultiAgentLearner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1).to(self.device) for key in self.group_keys}
        else:
            self.value_normalizer = None

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        buffer_size = self.config.buffer_size
        n_epochs = getattr(self.config, "n_epochs", 1)
        n_minibatch = getattr(self.config, "n_minibatch", 1)
        episode_length = self.episode_length
        if self.use_rnn:
            update_times = (self.config.running_steps // episode_length) // buffer_size
        else:
            update_times = self.config.running_steps // buffer_size
        total_iters = update_times * n_epochs * n_minibatch
        return total_iters

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def build_training_data(
            self,
            sample: Optional[dict],
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False
    ) -> OnPolicyMARLBatch:
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            OnPolicyMARLBatch: The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1

        obs_agent_wise = {
            agent: torch.as_tensor(sample['obs'][agent], device=self.device)
            for agent in self.agent_keys
        }
        act_agent_wise = {
            agent: torch.as_tensor(sample['actions'][agent], device=self.device)
            for agent in self.agent_keys
        }
        agent_mask_agent_wise = {
            agent: torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        values_agent_wise = {
            agent: torch.as_tensor(sample['values'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        returns_agent_wise = {
            agent: torch.as_tensor(sample['returns'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        advantages_agent_wise = {
            agent: torch.as_tensor(sample['advantages'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        log_pi_old_agent_wise = {
            agent: torch.as_tensor(sample['log_pi_old'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        avail_actions_agent_wise = None
        if use_actions_mask:
            avail_actions_agent_wise = {
                agent: torch.as_tensor(sample['avail_actions'][agent], device=self.device, dtype=torch.float32)
                for agent in self.agent_keys
            }
        state = None
        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)

        filled = None
        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        agent_indices = {}
        for group, n_agents in self.n_group_agents.items():
            agent_indices[group] = self.agent_indices[group].repeat(batch_size, 1).reshape(batch_size, n_agents, 1)
            if self.use_rnn:
                agent_indices[group] = agent_indices[group].unsqueeze(2).expand(-1, -1, seq_length, -1)

        return OnPolicyMARLBatch(
            batch_size=batch_size,
            global_states=state,
            observations=AgentGroupedTensor.from_agent_wise(obs_agent_wise, grouping=self.agent_grouping),
            actions=AgentGroupedTensor.from_agent_wise(act_agent_wise, grouping=self.agent_grouping),
            values=AgentGroupedTensor.from_agent_wise(values_agent_wise, grouping=self.agent_grouping),
            returns=AgentGroupedTensor.from_agent_wise(returns_agent_wise, grouping=self.agent_grouping),
            advantages=AgentGroupedTensor.from_agent_wise(advantages_agent_wise, grouping=self.agent_grouping),
            old_log_probs=AgentGroupedTensor.from_agent_wise(log_pi_old_agent_wise, grouping=self.agent_grouping),
            agent_masks=AgentGroupedTensor.from_agent_wise(agent_mask_agent_wise, grouping=self.agent_grouping),
            avail_actions=AgentGroupedTensor.from_agent_wise(avail_actions_agent_wise, grouping=self.agent_grouping),
            agent_indices=AgentGroupedTensor(agent_indices, grouping=self.agent_grouping),
            filled_masks=filled,
            seq_length=seq_length
        )


class OffPolicyMultiAgentLearner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(OffPolicyMultiAgentLearner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()
        self.mse_loss = nn.MSELoss()

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def build_training_data(
            self,
            sample: Optional[dict],
            use_actions_mask: Optional[bool] = False,
            use_global_state: Optional[bool] = False,
    ) -> OffPolicyMARLBatch:
        """
        Prepare the training data.

        Parameters:
            sample (dict): The raw sampled data.
            use_actions_mask (bool): Whether to use actions mask for unavailable actions.
            use_global_state (bool): Whether to use global state.

        Returns:
            OffPolicyMARLBatch: The formatted sampled data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1

        obs_agent_wise = {
            agent: torch.as_tensor(sample['obs'][agent], device=self.device)
            for agent in self.agent_keys
        }
        act_agent_wise = {
            agent: torch.as_tensor(sample['actions'][agent], device=self.device)
            for agent in self.agent_keys
        }
        if not self.use_rnn:
            obs_next_agent_wise = {
                agent: torch.as_tensor(sample['obs_next'][agent], device=self.device)
                for agent in self.agent_keys
            }
        else:
            obs_next_agent_wise = None
        rewards_agent_wise = {
            agent: torch.as_tensor(sample['rewards'][agent], device=self.device)
            for agent in self.agent_keys
        }
        terminals_agent_wise = {
            agent: torch.as_tensor(sample['terminals'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        agent_mask_agent_wise = {
            agent: torch.as_tensor(sample['agent_mask'][agent], device=self.device, dtype=torch.float32)
            for agent in self.agent_keys
        }
        avail_actions_agent_wise, avail_actions_next_agent_wise = None, None
        if use_actions_mask:
            avail_actions_agent_wise = {
                agent: torch.as_tensor(sample['avail_actions'][agent], device=self.device, dtype=torch.float32)
                for agent in self.agent_keys
            }
            if not self.use_rnn:
                avail_actions_next_agent_wise = {
                    agent: torch.as_tensor(sample['avail_actions_next'][agent], device=self.device, dtype=torch.float32)
                    for agent in self.agent_keys
                }
        state, state_next = None, None
        if use_global_state:
            state = torch.as_tensor(sample['state'], device=self.device)
            if not self.use_rnn:
                state_next = torch.as_tensor(sample['state_next'], device=self.device)

        filled = None
        if self.use_rnn:
            filled = torch.as_tensor(sample['filled'], device=self.device, dtype=torch.float32)

        agent_indices = {}
        for group, n_agents in self.n_group_agents.items():
            agent_indices[group] = self.agent_indices[group].repeat(batch_size, 1).reshape(batch_size, n_agents, 1)
            if self.use_rnn:
                agent_indices[group] = agent_indices[group].unsqueeze(2).expand(-1, -1, seq_length + 1, -1)

        return OffPolicyMARLBatch(
            batch_size=batch_size,
            global_states=state,
            next_global_states=state_next,
            observations=AgentGroupedTensor.from_agent_wise(obs_agent_wise, grouping=self.agent_grouping),
            actions=AgentGroupedTensor.from_agent_wise(act_agent_wise, grouping=self.agent_grouping),
            next_observations=AgentGroupedTensor.from_agent_wise(obs_next_agent_wise, grouping=self.agent_grouping),
            rewards=AgentGroupedTensor.from_agent_wise(rewards_agent_wise, grouping=self.agent_grouping),
            terminals=AgentGroupedTensor.from_agent_wise(terminals_agent_wise, grouping=self.agent_grouping),
            agent_masks=AgentGroupedTensor.from_agent_wise(agent_mask_agent_wise, grouping=self.agent_grouping),
            avail_actions=AgentGroupedTensor.from_agent_wise(avail_actions_agent_wise, grouping=self.agent_grouping),
            next_avail_actions=AgentGroupedTensor.from_agent_wise(avail_actions_next_agent_wise, self.agent_grouping),
            agent_indices=AgentGroupedTensor(agent_indices, self.agent_grouping),
            filled_masks=filled,
            seq_length=seq_length,
        )
