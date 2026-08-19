import gymnasium
import numpy as np
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.rl_models import (
    CategoricalActor, SAC_GaussianActor, TwinActionValueCritic, TwinDiscreteActionValueCritic)
from xuance.torch.rl_models.modules import ActionOutput
from xuance.torch.rl_models.architectures import SoftActorCritic, SoftActorCriticDiscrete


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
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
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
        critic = Critic(representation=deepcopy(representation),
                        action_space=self.action_space,
                        critic_hidden_size=self.config.critic_hidden_size,
                        normalizer=self.normalize_fn,
                        initializer=self.initializer,
                        activation=self.activation,
                        device=self.device)

        # build the RL model
        model = Architecture(actor=actor, critic=critic)

        return model

    def get_actions(
            self,
            observations: np.ndarray,
            test_mode: Optional[bool] = False
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
        actions_output = self.model.act(observations)
        actions = actions_output.detach().cpu().numpy()
        return ActionOutput(env_actions=actions)
