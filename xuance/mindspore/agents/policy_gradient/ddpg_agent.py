import numpy as np
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv

from xuance.mindspore import ms, ops, Tensor, Module
from xuance.mindspore.utils import ActivationFunctions
from xuance.mindspore.agents import OffPolicyAgent
from xuance.mindspore.rl_models.modules import ActionOutput
from xuance.mindspore.rl_models import DeterministicActor, ActionValueCritic
from xuance.mindspore.rl_models.architectures import DeterministicActorCritic


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
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=ActivationFunctions[self.config.activation_action]
        )

        # build critic network
        critic = ActionValueCritic(
            representation=deepcopy(representation),
            action_space=self.action_space,
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation
        )

        # build the RL model
        model = DeterministicActorCritic(actor=actor, critic=critic)

        return model

    def exploration(self, pi_actions: Tensor, *args, **kwargs) -> Tensor:
        explore_actions = (pi_actions + ops.randn_like(pi_actions) * self.noise_scale)
        explore_actions = ops.clamp(explore_actions, min=self.actions_low_tensor, max=self.actions_high_tensor)
        return explore_actions

    @ms.jit
    def _rollout_step(
            self,
            observations: Tensor,
            noise_scale: Tensor,
            **kwargs
    ) -> Tensor:
        pi_actions = self.model.act(observations)
        actions = (pi_actions + ops.randn_like(pi_actions) * self.noise_scale)
        actions = ops.clamp(actions, min=self.actions_low_tensor, max=self.actions_high_tensor)
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
        observations = Tensor(observations, dtype=ms.float32)
        noise_scale = Tensor(0.0 if test_mode else self.noise_scale, dtype=ms.float32)

        actions = self._rollout_step(observations, noise_scale=noise_scale)

        if not self.is_tensor_memory:
            actions = actions.numpy()

        return ActionOutput(env_actions=actions)

