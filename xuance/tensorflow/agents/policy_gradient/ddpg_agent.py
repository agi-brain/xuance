import numpy as np
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import tf, Tensor, Module
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents import OffPolicyAgent
from xuance.tensorflow.rl_models.modules import ActionOutput
from xuance.tensorflow.rl_models import DeterministicActor, ActionValueCritic
from xuance.tensorflow.rl_models.architectures import DeterministicActorCritic


class DDPG_Agent(OffPolicyAgent):
    """The implementation of DDPG agent.

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
        super(DDPG_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        self.model = self._build_model()  # build model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network
        actor = DeterministicActor(
            representation=representation,
            actor_hidden_size=self.config.actor_hidden_size,
            action_space=self.action_space,
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=ActivationFunctions[self.config.activation_action]
        )

        # build critic network
        critic = ActionValueCritic(
            representation=representation.clone(copy_weights=False, trainable=True, name="critic_representation"),
            action_space=self.action_space,
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation
        )

        # build the RL model
        model = DeterministicActorCritic(actor=actor, critic=critic)

        return model

    @tf.function(reduce_retracing=True)
    def _rollout_step(
            self,
            observations: Tensor,
            noise_scale: Tensor,
            **kwargs
    ) -> Tensor:
        pi_actions = self.model(observations).actions

        noise = tf.random.normal(shape=tf.shape(pi_actions), dtype=pi_actions.dtype)
        actions = pi_actions + noise * tf.cast(noise_scale, pi_actions.dtype)
        actions = tf.clip_by_value(actions,
                                   tf.cast(self.actions_low_tensor, pi_actions.dtype),
                                   tf.cast(self.actions_high_tensor, pi_actions.dtype))
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
        noise_scale = tf.convert_to_tensor(0.0 if test_mode else self.noise_scale, dtype=tf.float32)

        actions = self._rollout_step(observations, noise_scale=noise_scale)

        if not self.is_tensor_memory:
            actions = actions.numpy()

        return ActionOutput(env_actions=actions)
