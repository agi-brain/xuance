import numpy as np
from tqdm import tqdm
from argparse import Namespace
from xuance.common import Optional, DummyOffPolicyBuffer, OfflineBuffer_D4RL
from xuance.torch import Module
from xuance.torch.agents.base import Agent, BaseCallback


class OfflineAgent(Agent):
    """The core class for offline reinforcement learning.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
                It can be used for logging, early stopping, model saving, or visualization.
                If not provided, a default no-op callback is used.

    """
    def __init__(self,
                 config: Namespace,
                 envs,
                 callback: Optional[BaseCallback] = None):
        super(OfflineAgent, self).__init__(config, envs, callback)
        self.auxiliary_info_shape = None
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size
        self.memory: Optional[OfflineBuffer_D4RL] = self._build_memory()

    def _build_memory(self, auxiliary_info_shape=None):
        self.d4rl = True if self.config.env_name != "atari" else False
        Buffer = OfflineBuffer_D4RL if self.d4rl else DummyOffPolicyBuffer
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        return train_info

    def train(self, train_steps):
        train_info = {}
        for _ in tqdm(range(train_steps)):
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)  # self.n_epochs = 16
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  train_info=train_info, train_steps=train_steps)
            self.current_step += 1
        return train_info

    def action(self, observations: np.ndarray):
        raise NotImplementedError

    def test(self, env_fn, steps):
        raise NotImplementedError
