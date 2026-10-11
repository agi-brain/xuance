# This is the main file for an advantage actor critic (A2C) algorithm.
# The agent random sample a batch in the replay buffer, and optimize the policy gradient and value function loss.
# This can be a first RL algorithm code for the starters.
import gymnasium
import numpy as np
from argparse import Namespace
from copy import deepcopy
from gymnasium.spaces import Space
from typing import Optional, Tuple
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.mindspore import ms, ops, Tensor, Module
from xuance.mindspore.utils import ActivationFunctions
from xuance.mindspore.agents import OnPolicyAgent
from xuance.mindspore.rl_models import CategoricalActor, GaussianActor
from xuance.mindspore.rl_models import StateValueCritic as Critic
from xuance.mindspore.rl_models import ActorCritic
from xuance.mindspore.rl_models.modules import ActionOutput


class A2C_Agent(OnPolicyAgent):
    """The implementation of A2C agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
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
        super(A2C_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network
        actor_input = dict(
            representation=representation,
            actor_hidden_size=self.config.actor_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation
        )
        if isinstance(self.action_space, gymnasium.spaces.Box):
            Actor = GaussianActor
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            Actor = CategoricalActor
        else:
            raise NotImplementedError
        actor = Actor(**actor_input)

        # build critic network
        critic = Critic(
            representation=deepcopy(representation),
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation
        )

        # build the RL model
        model = ActorCritic(actor=actor, critic=critic)

        return model

    @ms.jit
    def _rollout_step(
            self,
            observations: Tensor
    ) -> Tuple[Tensor, ...]:
        if self.model.is_continuous:
            a_mean, log_std, values = self.model(observations)
            a_std = ops.exp(log_std)
            eps = ops.normal(shape=a_mean.shape, mean=Tensor(0.0), stddev=Tensor(1.0))  # 𝜖 ~ N(0, 1)
            action_sampled = a_mean + a_std * eps
        else:
            logits, values = self.model(observations)
            pi_dists = ops.RandomCategorical(ms.int32)
            action_sampled = pi_dists(logits, 1, 0).squeeze(-1)
        return action_sampled, values

    def get_actions(self, observations: np.ndarray, **kwargs) -> ActionOutput:
        observations = Tensor(observations, dtype=ms.float32)
        actions, values = self._rollout_step(observations)

        if not self.is_tensor_memory:
            actions = actions.numpy()
            values = values.numpy()

        return ActionOutput(
            env_actions=actions,
            values=values
        )
