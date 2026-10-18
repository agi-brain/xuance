import os
from abc import abstractmethod
from argparse import Namespace
from xuance.common import Optional, AgentGrouping

from xuance.mindspore import ms, ops, nn, Module, Tensor, optim
from xuance.mindspore.utils import ValueNorm, AgentGroupedTensor
from xuance.mindspore.rl_models.modules import OnPolicyMARLBatch, OffPolicyMARLBatch
from xuance.mindspore.learners.base.drl_learner import Learner


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
        self.agent_indices = {k: Tensor(self.agent_grouping.agent_indices(k), dtype=ms.int64)
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
            joint_tensor = ops.concat([input_tensor[key] for key in self.agent_keys], axis=-1)
        if output_shape is not None:
            joint_tensor = joint_tensor.reshape(output_shape)
        return joint_tensor

    def build_training_data(self,
                            sample: Optional[dict],
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False) -> OnPolicyMARLBatch | OffPolicyMARLBatch:
        raise NotImplementedError

    def build_optimizer(self):
        self.optimizer = optim.Adam(
            params=self.model.trainable_params(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.end_factor_lr_decay,
            total_iters=self.config.running_steps
        )

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_path):
        model_path += ".pth"
        ms.save_checkpoint(self.model, model_path)

    def load_model(self, path, model=None):
        path = os.path.join(path, model) if model is not None else path
        path += ".pth"
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
        ms.load_param_into_net(self.model, ms.load_checkpoint(model_path))
        print(f"Successfully load model from '{path}'.")


class OnPolicyMultiAgentLearner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
                 callback):
        super(OnPolicyMultiAgentLearner, self).__init__(config, agent_grouping, model, callback)
        self.build_optimizer()

        self.use_value_clip = config.use_value_clip
        self.value_clip_range = config.value_clip_range
        self.use_huber_loss = config.use_huber_loss
        self.huber_delta = config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef

        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)

        if self.use_value_norm:
            self.value_normalizer = {key: ValueNorm(1) for key in self.group_keys}
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
            agent: Tensor(sample['obs'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        act_agent_wise = {
            agent: Tensor(sample['actions'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        agent_mask_agent_wise = {
            agent: Tensor(sample['agent_mask'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        values_agent_wise = {
            agent: Tensor(sample['values'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        returns_agent_wise = {
            agent: Tensor(sample['returns'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        advantages_agent_wise = {
            agent: Tensor(sample['advantages'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        log_pi_old_agent_wise = {
            agent: Tensor(sample['log_pi_old'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        avail_actions_agent_wise = None
        if use_actions_mask:
            avail_actions_agent_wise = {
                agent: Tensor(sample['avail_actions'][agent], dtype=ms.float32)
                for agent in self.agent_keys
            }
        state = None
        if use_global_state:
            state = Tensor(sample['state'], dtype=ms.float32)

        filled = None
        if self.use_rnn:
            filled = Tensor(sample['filled'], dtype=ms.float32)

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
            agent: Tensor(sample['obs'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        act_agent_wise = {
            agent: Tensor(sample['actions'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        if not self.use_rnn:
            obs_next_agent_wise = {
                agent: Tensor(sample['obs_next'][agent], dtype=ms.float32)
                for agent in self.agent_keys
            }
        else:
            obs_next_agent_wise = None
        rewards_agent_wise = {
            agent: Tensor(sample['rewards'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        terminals_agent_wise = {
            agent: Tensor(sample['terminals'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        agent_mask_agent_wise = {
            agent: Tensor(sample['agent_mask'][agent], dtype=ms.float32)
            for agent in self.agent_keys
        }
        avail_actions_agent_wise, avail_actions_next_agent_wise = None, None
        if use_actions_mask:
            avail_actions_agent_wise = {
                agent: Tensor(sample['avail_actions'][agent], dtype=ms.float32)
                for agent in self.agent_keys
            }
            if not self.use_rnn:
                avail_actions_next_agent_wise = {
                    agent: Tensor(sample['avail_actions_next'][agent], dtype=ms.float32)
                    for agent in self.agent_keys
                }
        state, state_next = None, None
        if use_global_state:
            state = Tensor(sample['state'], dtype=ms.float32)
            if not self.use_rnn:
                state_next = Tensor(sample['state_next'], dtype=ms.float32)

        filled = None
        if self.use_rnn:
            filled = Tensor(sample['filled'], dtype=ms.float32)

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
