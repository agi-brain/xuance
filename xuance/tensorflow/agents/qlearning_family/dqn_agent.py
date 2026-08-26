import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from typing import Optional, Tuple
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import tf, Tensor, Module
from xuance.tensorflow.agents import OffPolicyAgent
from xuance.tensorflow.rl_models.modules import ActionOutput
from xuance.tensorflow.rl_models.architectures import DeepQNetwork


class DQN_Agent(OffPolicyAgent):
    """The implementation of Deep Q-Networks (DQN) agent.

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
        super(DQN_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build the RL model.
        model = DeepQNetwork(
            representation=representation,
            hidden_size=self.config.q_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation,
            use_distributed_training=self.distributed_training
        )

        return model

    @tf.function(reduce_retracing=True)
    def _rollout_step(
            self, observations: Tensor, epsilon: Tensor, **kwargs
    ) -> Tuple[Tensor, ...]:
        greedy_actions = self.model(observations).actions
        explore_mask = tf.random.uniform(shape=tf.shape(greedy_actions),
                                         minval=0.0,
                                         maxval=1.0,
                                         dtype=tf.float32) < epsilon
        random_actions = tf.random.uniform(shape=tf.shape(greedy_actions),
                                           minval=0,
                                           maxval=self.action_space.n,
                                           dtype=greedy_actions.dtype)
        actions = tf.where(explore_mask, random_actions, greedy_actions)

        return actions

    def get_actions(
            self,
            observations: np.ndarray | Tensor,
            test_mode: bool = False
    ) -> ActionOutput:
        """Returns actions for the given observations.

        Args:
            observations: Observations used by the policy to generate actions.
            test_mode: Whether to disable exploration noise for evaluation.

        Returns:
            The ActionOutput containing actions to be executed in the environment.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        epsilon = tf.convert_to_tensor(0.0 if test_mode else self.e_greedy, dtype=tf.float32)

        actions = self._rollout_step(observations, epsilon=epsilon)

        if not self.is_tensor_memory:
            actions = actions.numpy()

        return ActionOutput(env_actions=actions)
