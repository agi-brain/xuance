from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents.policy_gradient.ddpg_agent import DDPG_Agent
from xuance.torch.rl_models import DeterministicActor, TwinActionValueCritic
from xuance.torch.rl_models.architectures import TwinDelayedActorCritic


class TD3_Agent(DDPG_Agent):
    """The implementation of TD3 agent.

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
        super(TD3_Agent, self).__init__(config, envs, observation_space, action_space, callback)

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
            activation_action=ActivationFunctions[self.config.activation_action],
            device=self.device
        )

        # build critic network
        critic = TwinActionValueCritic(
            representation=deepcopy(representation),
            action_space=self.action_space,
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )

        # build the RL model
        model = TwinDelayedActorCritic(
            actor=actor,
            critic=critic,
            target_policy_noise=getattr(self.config, "target_policy_noise", 0.2),
            target_noise_clip=getattr(self.config, "target_noise_clip", 0.5)
        )

        return model
