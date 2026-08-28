import gymnasium
import numpy as np
from argparse import Namespace

from gymnasium.spaces import Space
from typing import Optional
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import tf, Tensor, Module
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents import OffPolicyAgent
from xuance.tensorflow.rl_models import (CategoricalActor, SAC_GaussianActor,
                                         TwinActionValueCritic, TwinDiscreteActionValueCritic)
from xuance.tensorflow.rl_models.modules import ActionOutput
from xuance.tensorflow.rl_models.architectures import SoftActorCritic, SoftActorCriticDiscrete


class SAC_Agent(OffPolicyAgent):
    """The implementation of SAC agent.

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
        super(SAC_Agent, self).__init__(config, envs, observation_space, action_space, callback)

        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)

    def _build_model(self) -> Module:
        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network
        actor_input = dict(
            representation=representation,
            actor_hidden_size=self.config.actor_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation
        )
        if isinstance(self.action_space, gymnasium.spaces.Box):
            Actor = SAC_GaussianActor
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
            Critic = TwinActionValueCritic
            Architecture = SoftActorCritic
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            Actor = CategoricalActor
            Critic = TwinDiscreteActionValueCritic
            Architecture = SoftActorCriticDiscrete
        else:
            raise NotImplementedError
        actor = Actor(**actor_input)

        # build critic network
        critic = Critic(representation=representation.clone(copy_weights=False, trainable=True,
                                                            name="critic_representation"),
                        action_space=self.action_space,
                        critic_hidden_size=self.config.critic_hidden_size,
                        normalizer=self.normalizer_fn,
                        initializer=self.initializer,
                        activation=self.activation)

        # build the RL model
        model = Architecture(actor=actor, critic=critic)

        return model

    @tf.function
    def _stochastic_rollout_step(self, observations: Tensor, **kwargs) -> Tensor:
        actions = self.model(observations, deterministic=False).actions
        return actions

    @tf.function
    def _deterministic_rollout_step(self, observations: Tensor, **kwargs) -> Tensor:
        actions = self.model(observations, deterministic=True).actions
        return actions

    def get_actions(
            self,
            observations: np.ndarray,
            test_mode: Optional[bool] = False,
            deterministic: Optional[bool] = False
    ) -> ActionOutput:
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)

        if deterministic:
            actions = self._deterministic_rollout_step(observations)
        else:
            actions = self._stochastic_rollout_step(observations)

        if not self.is_tensor_memory:
            actions = actions.numpy()

        return ActionOutput(env_actions=actions)
