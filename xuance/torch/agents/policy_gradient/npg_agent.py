import gymnasium
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


class NPG_Agent(OnPolicyAgent):
    """The implementation of NPG agent.

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
        super(NPG_Agent, self).__init__(config, envs, observation_space, action_space, callback)
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
            activation=self.activation,
            device=self.device
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
        critic = Critic(representation=deepcopy(representation),
                        critic_hidden_size=self.config.critic_hidden_size,
                        normalizer=self.normalize_fn,
                        initializer=self.initializer,
                        activation=self.activation,
                        device=self.device)
        # build the RL model
        model = ActorCritic(actor=actor, critic=critic)
        return model
