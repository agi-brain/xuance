import os
import re
from pathlib import Path
from abc import abstractmethod
from argparse import Namespace
from xuance.common import Optional, AgentGrouping

from xuance.tensorflow import tf, Module
from xuance.tensorflow.utils import ValueNorm, AgentGroupedTensor
from xuance.tensorflow.rl_models.modules import OnPolicyMARLBatch, OffPolicyMARLBatch
from xuance.tensorflow.learners.base.drl_learner import Learner


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
        with tf.device(self.device):
            self.agent_indices = {
                k: tf.convert_to_tensor(self.agent_grouping.agent_indices(k), dtype=tf.int64)
                for k in self.group_keys
            }

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
            joint_tensor = tf.concat([input_tensor[key] for key in self.agent_keys], axis=-1)
        if output_shape is not None:
            joint_tensor = tf.reshape(joint_tensor, output_shape)
        return joint_tensor

    def build_training_data(self,
                            sample: Optional[dict],
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False) -> OnPolicyMARLBatch | OffPolicyMARLBatch:
        raise NotImplementedError

    def build_optimizer(self):
        weight_decay = getattr(self.config, "weight_decay", 0.0)

        # Equivalent to PyTorch LinearLR:
        #   start_factor = 1.0
        #   end_factor = self.end_factor_lr_decay
        #   total_iters = self.total_iters
        self.scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=max(int(self.total_iters), 1),
            end_learning_rate=self.learning_rate * self.end_factor_lr_decay,
            power=1.0,
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.scheduler,
            epsilon=1e-5,
            weight_decay=weight_decay,
        )

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _safe_checkpoint_name(name: str) -> str:
        """Convert an arbitrary dictionary key to a Checkpoint-safe name."""
        name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
        if not name:
            name = "item"
        if name[0].isdigit():
            name = f"item_{name}"
        return name

    def _checkpoint_items(self):
        """
        Build a flat dictionary of TensorFlow Trackable objects.

        TensorFlow Checkpoint does not use PyTorch-style state_dict objects.
        Nested optimizer dictionaries are flattened into stable names so that
        model and optimizer states can be restored together.
        """
        items = {"policy": self.model}

        if self.optimizer is None:
            return items

        if isinstance(self.optimizer, dict):
            for key, value in self.optimizer.items():
                key_safe = self._safe_checkpoint_name(key)

                if isinstance(value, dict):
                    for sub_key, optimizer in value.items():
                        sub_key_safe = self._safe_checkpoint_name(sub_key)
                        items[f"optimizer_{key_safe}_{sub_key_safe}"] = optimizer
                else:
                    items[f"optimizer_{key_safe}"] = value
        else:
            items["optimizer"] = self.optimizer

        return items

    def save_model(self, model_path):
        """
        Save model and optimizer states using tf.train.Checkpoint.

        Notes:
            `model_path` is treated as a TensorFlow checkpoint prefix. TensorFlow
            creates `<model_path>.index` and one or more
            `<model_path>.data-*` files.
        """
        model_path = str(model_path)
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        checkpoint = tf.train.Checkpoint(**self._checkpoint_items())
        checkpoint.write(model_path)

    def _find_checkpoint(self, path, model=None):
        """
        Resolve a TensorFlow checkpoint prefix from a file/prefix or seed folder.
        """
        target_path = os.path.join(path, model) if model is not None else path

        # TensorFlow checkpoints are represented by a prefix plus .index/.data.
        if os.path.isfile(target_path + ".index"):
            return target_path, os.path.dirname(target_path)

        if target_path.endswith(".index") and os.path.isfile(target_path):
            prefix = target_path[:-len(".index")]
            return prefix, os.path.dirname(prefix)

        if not os.path.isdir(path):
            raise RuntimeError(
                f"The path '{path}' is not a valid directory or checkpoint prefix!"
            )

        folder_names = sorted(
            f for f in os.listdir(path)
            if "seed_" in f and os.path.isdir(os.path.join(path, f))
        )
        if not folder_names:
            raise RuntimeError(
                f"No model folders with 'seed_' found in '{path}'!"
            )

        seed_path = Path(path) / folder_names[-1]
        dir_name = str(seed_path)

        index_files = list(seed_path.glob("*.index"))
        if not index_files:
            raise FileNotFoundError(
                f"No TensorFlow checkpoint (.index) file found in {seed_path}"
            )

        # Prefer the final training checkpoint when present.
        preferred_names = (
            "final_train_model.index",
            "final_train_model.ckpt.index",
        )
        model_index = None
        for name in preferred_names:
            candidate = seed_path / name
            if candidate.exists():
                model_index = candidate
                break

        if model_index is None:
            # Fall back to the most recently modified checkpoint.
            model_index = max(index_files, key=lambda p: p.stat().st_mtime)

        return str(model_index)[:-len(".index")], dir_name

    def load_model(self, path, model=None):
        model_path, dir_name = self._find_checkpoint(path, model)

        checkpoint = tf.train.Checkpoint(**self._checkpoint_items())
        status = checkpoint.restore(model_path)

        # Optimizer slot variables may be created lazily. `expect_partial()`
        # allows restoration before all such variables have necessarily been
        # materialized, while model variables are restored by TensorFlow's
        # deferred restoration mechanism.
        status.expect_partial()

        self.learning_rate = self._get_current_learning_rate()

        print(f"Successfully load model from '{model_path}'.")
        return dir_name

    def _get_current_learning_rate(self):
        """Return the current scalar learning rate."""
        if self.optimizer is None:
            return self.learning_rate

        optimizer = self.optimizer
        if isinstance(optimizer, dict):
            optimizer = next(iter(optimizer.values()))
            if isinstance(optimizer, dict):
                optimizer = next(iter(optimizer.values()))

        lr = optimizer.learning_rate

        if callable(lr):
            lr = lr(optimizer.iterations)

        if isinstance(lr, tf.Variable):
            lr = lr.read_value()

        return float(tf.convert_to_tensor(lr).numpy())


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

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.huber_loss = tf.keras.losses.Huber(delta=self.huber_delta,
                                                reduction=tf.keras.losses.Reduction.NONE)
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

        with tf.device(self.device):
            obs_agent_wise = {
                agent: tf.convert_to_tensor(sample['obs'][agent])
                for agent in self.agent_keys
            }
            act_agent_wise = {
                agent: tf.convert_to_tensor(sample['actions'][agent])
                for agent in self.agent_keys
            }
            agent_mask_agent_wise = {
                agent: tf.convert_to_tensor(sample['agent_mask'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            values_agent_wise = {
                agent: tf.convert_to_tensor(sample['values'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            returns_agent_wise = {
                agent: tf.convert_to_tensor(sample['returns'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            advantages_agent_wise = {
                agent: tf.convert_to_tensor(sample['advantages'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            log_pi_old_agent_wise = {
                agent: tf.convert_to_tensor(sample['log_pi_old'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            avail_actions_agent_wise = None
            if use_actions_mask:
                avail_actions_agent_wise = {
                    agent: tf.convert_to_tensor(sample['avail_actions'][agent], dtype=tf.float32)
                    for agent in self.agent_keys
                }
            state = None
            if use_global_state:
                state = tf.convert_to_tensor(sample['state'])

            filled = None
            if self.use_rnn:
                filled = tf.convert_to_tensor(sample['filled'], dtype=tf.float32)

            agent_indices = {}
            for group, n_agents in self.n_group_agents.items():
                indices = tf.tile(self.agent_indices[group][None, :], [batch_size, 1])
                indices = tf.reshape(indices, [batch_size, n_agents, 1])
                if self.use_rnn:
                    indices = tf.expand_dims(indices, axis=2)
                    indices = tf.broadcast_to(indices, [batch_size, n_agents, seq_length, 1])
                agent_indices[group] = indices

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
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def learn(self, **kwargs):
        if self.distributed_training:
            info_train = self.model.mirrored_strategy.run(self.forward_fn, kwargs=kwargs)
            return info_train[0]
        else:
            return self.forward_fn(**kwargs)

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

        with tf.device(self.device):
            obs_agent_wise = {
                agent: tf.convert_to_tensor(sample['obs'][agent])
                for agent in self.agent_keys
            }
            act_agent_wise = {
                agent: tf.convert_to_tensor(sample['actions'][agent])
                for agent in self.agent_keys
            }
            if not self.use_rnn:
                obs_next_agent_wise = {
                    agent: tf.convert_to_tensor(sample['obs_next'][agent])
                    for agent in self.agent_keys
                }
            else:
                obs_next_agent_wise = None
            rewards_agent_wise = {
                agent: tf.convert_to_tensor(sample['rewards'][agent])
                for agent in self.agent_keys
            }
            terminals_agent_wise = {
                agent: tf.convert_to_tensor(sample['terminals'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            agent_mask_agent_wise = {
                agent: tf.convert_to_tensor(sample['agent_mask'][agent], dtype=tf.float32)
                for agent in self.agent_keys
            }
            avail_actions_agent_wise, avail_actions_next_agent_wise = None, None
            if use_actions_mask:
                avail_actions_agent_wise = {
                    agent: tf.convert_to_tensor(sample['avail_actions'][agent], dtype=tf.float32)
                    for agent in self.agent_keys
                }
                if not self.use_rnn:
                    avail_actions_next_agent_wise = {
                        agent: tf.convert_to_tensor(sample['avail_actions_next'][agent], dtype=tf.float32)
                        for agent in self.agent_keys
                    }
            state, state_next = None, None
            if use_global_state:
                state = tf.convert_to_tensor(sample['state'])
                if not self.use_rnn:
                    state_next = tf.convert_to_tensor(sample['state_next'])

            filled = None
            if self.use_rnn:
                filled = tf.convert_to_tensor(sample['filled'], dtype=tf.float32)

            agent_indices = {}
            for group, n_agents in self.n_group_agents.items():
                indices = tf.tile(self.agent_indices[group][None, :], [batch_size, 1])
                indices = tf.reshape(indices, [batch_size, n_agents, 1])

                if self.use_rnn:
                    indices = tf.expand_dims(indices, axis=2)
                    indices = tf.broadcast_to(indices, [batch_size, n_agents, seq_length + 1, 1])

                agent_indices[group] = indices

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
