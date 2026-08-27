# This is the main file for an advantage actor critic (A2C) algorithm.
# The agent random sample a batch in the replay buffer, and optimize the policy gradient and value function loss.
# This can be a first RL algorithm code for the starters.
import gymnasium
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from typing import Optional, Tuple
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import tf, Tensor, Module
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents import OnPolicyAgent
from xuance.tensorflow.rl_models.modules import ActionOutput
from xuance.tensorflow.rl_models import CategoricalActor, GaussianActor, ActorCritic
from xuance.tensorflow.rl_models import StateValueCritic as Critic


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
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation,
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
        critic = Critic(representation=representation.clone(copy_weights=False, trainable=True,
                                                            name="critic_representation"),
                        critic_hidden_size=self.config.critic_hidden_size,
                        normalizer=self.normalizer_fn,
                        initializer=self.initializer,
                        activation=self.activation)

        # build the RL model
        model = ActorCritic(actor=actor, critic=critic)

        return model

    @tf.function
    def _stochastic_rollout_step(
            self,
            observations: Tensor,
            **kwargs
    ) -> Tuple[Tensor, ...]:
        model_output = self.model(observations)
        policy_dists = model_output.distributions
        actions = policy_dists.stochastic_sample()
        values = model_output.values
        return actions, values

    @tf.function
    def _deterministic_rollout_step(
            self,
            observations: Tensor,
            **kwargs
    ) -> Tuple[Tensor, ...]:
        model_output = self.model(observations)
        policy_dists = model_output.distributions
        actions = policy_dists.deterministic_sample()
        values = model_output.values
        return actions, values

    def get_actions(
            self,
            observations: np.ndarray,
            deterministic: bool = False,
            **kwargs
    ) -> ActionOutput:
        """Compute actions and value estimates for a batch of observations.

        This method performs a forward pass through the current policy to obtain action distributions
        and value predictions. Actions are sampled stochastically from the policy distribution.

        Args:
            observations (np.ndarray): Batch of observations. The array is expected to have shape compatible with
                the underlying policy.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            ActionOutput.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)

        if deterministic:
            actions, values = self._deterministic_rollout_step(observations)
        else:
            actions, values = self._stochastic_rollout_step(observations)

        if self.is_tensor_memory:
            values = 0 if values is None else values
        else:
            actions = actions.numpy()
            values = 0 if values is None else values.numpy()
        return ActionOutput(
            env_actions=actions,
            values=values,
        )


