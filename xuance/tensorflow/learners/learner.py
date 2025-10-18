import os
import platform
import numpy as np
from operator import itemgetter
from abc import ABC, abstractmethod
from argparse import Namespace
from xuance.common import Union, List, Optional
from xuance.tensorflow import Module, tk, tf


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: Module,
                 callback):
        self.os_name = platform.platform()
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training

        self.episode_length = config.episode_length
        self.learning_rate = config.learning_rate if hasattr(config, 'learning_rate') else None
        self.use_linear_lr_decay = config.use_linear_lr_decay if hasattr(config, 'use_linear_lr_decay') else False
        self.end_factor_lr_decay = config.end_factor_lr_decay if hasattr(config, 'end_factor_lr_decay') else 0.5
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer: Union[dict, list, Optional[tk.optimizers.Optimizer]] = None
        self.callback = callback

        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

    def save_model(self, model_path):
        self.policy.save_weights(model_path)

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
        latest = tf.train.latest_checkpoint(model_path)
        try:
            self.policy.load_weights(latest)
        except:
            raise "Failed to load model! Please train and save the model first."

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: Module,
                 callback):
        self.os_name = platform.platform()
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.use_parameter_sharing = config.use_parameter_sharing
        self.model_keys = model_keys
        self.agent_keys = agent_keys
        self.episode_length = config.episode_length
        self.learning_rate = getattr(config, 'learning_rate', None)
        self.use_linear_lr_decay = getattr(config, 'use_linear_lr_decay', False)
        self.end_factor_lr_decay = getattr(config, 'end_factor_lr_decay', 0.5)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.use_rnn = getattr(config, 'use_rnn', False)
        self.use_actions_mask = getattr(config, 'use_actions_mask', False)
        self.policy = policy
        self.optimizer: Union[dict, list, Optional[tk.optimizers]] = None
        self.callback = callback

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
            obs_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)
            actions_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)
            rewards_tensor = tf.stack(itemgetter(*self.agent_keys)(sample['rewards']), axis=1)
            ter_tensor = tf.cast(tf.stack(itemgetter(*self.agent_keys)(sample['terminals']), axis=1), dtype=tf.float32)
            msk_tensor = tf.cast(tf.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), axis=1), dtype=tf.float32)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape([bs, seq_length + 1, -1])}
                if len(actions_tensor.shape) == 3:
                    actions = {k: tf.reshape(actions_tensor, [bs, seq_length])}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: tf.reshape(actions_tensor, [bs, seq_length, -1])}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: tf.reshape(rewards_tensor, [batch_size, self.n_agents, seq_length])}
                terminals = {k: tf.reshape(ter_tensor, [batch_size, self.n_agents, seq_length])}
                agent_mask = {k: tf.reshape(msk_tensor, [bs, seq_length])}
                IDs = tf.reshape(tf.tile(tf.eye(self.n_agents, dtype=np.float32)[None, :, None, :],
                                         [batch_size, 1, seq_length + 1, 1]), [bs, seq_length + 1, self.n_agents])
            else:
                obs = {k: tf.reshape(obs_tensor, [bs, -1])}
                if len(actions_tensor.shape) == 2:
                    actions = {k: tf.reshape(actions_tensor, [bs])}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: tf.reshape(actions_tensor, [bs, -1])}
                else:
                    raise AttributeError("Wrong actions shape.")
                rewards = {k: tf.reshape(rewards_tensor, [batch_size, self.n_agents])}
                terminals = {k: tf.reshape(ter_tensor, [batch_size, self.n_agents])}
                agent_mask = {k: tf.reshape(msk_tensor, [bs])}
                obs_next = {k: tf.reshape(tf.stack(itemgetter(*self.agent_keys)(sample['obs_next']), axis=1), [bs, -1])}
                IDs = tf.reshape(tf.tile(tf.eye(self.n_agents, dtype=np.float32)[None],
                                         [batch_size, 1, 1]), [bs, self.n_agents])

            if use_actions_mask:
                avail_a = tf.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: tf.cast(tf.reshape(avail_a, [bs, seq_length + 1, -1]), dtype=tf.float32)}
                else:
                    avail_actions = {k: tf.cast(tf.reshape(avail_a, [bs, -1]), dtype=tf.float32)}
                    avail_a_next = tf.stack(itemgetter(*self.agent_keys)(sample['avail_actions_next']), axis=1)
                    avail_actions_next = {k: tf.cast(tf.reshape(avail_a_next, [bs, -1]), dtype=tf.float32)}
        else:
            obs = {k: tf.convert_to_tensor(sample['obs'][k], dtype=tf.float32) for k in self.agent_keys}
            actions = {k: tf.convert_to_tensor(sample['actions'][k], dtype=tf.float32) for k in self.agent_keys}
            rewards = {k: tf.convert_to_tensor(sample['rewards'][k], dtype=tf.float32) for k in self.agent_keys}
            terminals = {k: tf.convert_to_tensor(sample['terminals'][k], dtype=tf.float32) for k in self.agent_keys}
            agent_mask = {k: tf.convert_to_tensor(sample['agent_mask'][k], dtype=tf.float32) for k in self.agent_keys}
            if not self.use_rnn:
                obs_next = {k: tf.convert_to_tensor(sample['obs_next'][k], dtype=tf.float32) for k in self.agent_keys}
            if use_actions_mask:
                avail_actions = {k: tf.convert_to_tensor(sample['avail_actions'][k], dtype=tf.float32)
                                 for k in self.agent_keys}
                if not self.use_rnn:
                    avail_actions_next = {k: tf.convert_to_tensor(sample['avail_actions_next'][k], dtype=tf.float32)
                                          for k in self.model_keys}

        if use_global_state:
            state = tf.convert_to_tensor(sample['state'], dtype=tf.float32)
            if not self.use_rnn:
                state_next = tf.convert_to_tensor(sample['state_next'], dtype=tf.float32)

        if self.use_rnn:
            filled = tf.convert_to_tensor(sample['filled'], dtype=tf.float32)

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
        self.policy.save_weights(model_path)

    def load_model(self, path, seed=1):
        try: file_names = os.listdir(path)
        except: raise "Failed to load model! Please train and save the model first."
        model_path = ''

        for f in file_names:
            '''Change directory to the specified seed (if exists)'''
            if f"seed_{seed}" in f:
                model_path = os.path.join(path, f)
                if os.listdir(model_path).__len__() == 0:
                    continue
                else:
                    break
        if model_path == '':
            raise RuntimeError("Failed to load model! Please train and save the model first.")
        latest = tf.train.latest_checkpoint(model_path)
        try:
            self.policy.load_weights(latest)
        except:
            raise RuntimeError("Failed to load model! Please train and save the model first.")
