import gymnasium
import numpy as np
from argparse import Namespace
from copy import deepcopy
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OnPolicyAgent
from xuance.torch.rl_models import CategoricalActor, GaussianActor
from xuance.torch.rl_models import StateValueCritic as Critic
from xuance.torch.rl_models import ActorCritic


class TRPO_Agent(OnPolicyAgent):
    """The implementation of Trust Region Policy Optimization (TRPO) agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        observation_space: optional observation space when envs is None.
        action_space: optional action space when envs is None.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(TRPO_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network (kept separate from critic so critic updates do not violate the trust region)
        actor_input = dict(
            representation=representation,
            actor_hidden_size=self.config.actor_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )
        if isinstance(self.action_space, gymnasium.spaces.Box):
            Actor = GaussianActor
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            Actor = CategoricalActor
        else:
            raise NotImplementedError(f"TRPO does not support action space {type(self.action_space)}.")
        actor = Actor(**actor_input)

        # build critic network
        critic = Critic(representation=deepcopy(representation),
                        critic_hidden_size=self.config.critic_hidden_size,
                        normalizer=self.normalize_fn,
                        initializer=self.initializer,
                        activation=self.activation,
                        device=self.device)

        # build the RL model
        model = ActorCritic(actor=actor, critic=critic)
        return model

    def train_epochs(self, n_epochs: int = 1) -> dict:
        """Perform one full-batch trust-region actor update followed by mini-batch critic updates.

        Args:
            n_epochs (int): Number of critic optimization epochs over the collected rollout buffer.

        Returns:
            dict: Aggregated training metrics from the actor and critic updates.
        """
        indexes = np.arange(self.buffer_size)
        train_info = {}

        # 1. Single full-batch trust-region policy update under the current rollout policy.
        full_samples = self.memory.sample(indexes)
        actor_info = self.learner.update_actor(**full_samples)
        train_info.update(actor_info)

        # 2. Multi-epoch mini-batch value function updates.
        critic_info = {}
        for _ in range(n_epochs):
            np.random.shuffle(indexes)
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                sample_idx = indexes[start:end]
                samples = self.memory.sample(sample_idx)
                critic_info = self.learner.update_critic(**samples)
        train_info.update(critic_info)
        return train_info
