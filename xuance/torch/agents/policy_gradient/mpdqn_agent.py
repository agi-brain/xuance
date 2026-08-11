import numpy as np
from copy import deepcopy
from argparse import Namespace
from gymnasium import spaces
from xuance.environment.single_agent_env import Gym_Env
from xuance.common import Optional, BaseCallback
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents.policy_gradient.pdqn_agent import PDQN_Agent
from xuance.torch.rl_models import DeterministicActor, HybridActionValueCritic
from xuance.torch.rl_models.architectures import MultipassParameterizedDQN


class MPDQN_Agent(PDQN_Agent):
    """The implementation of MPDQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 callback: Optional[BaseCallback] = None):
        super(MPDQN_Agent, self).__init__(config, envs, callback)

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build the RL model.
        continuous_actor = DeterministicActor(
            representation=representation,
            actor_hidden_size=self.config.conactor_hidden_size,
            action_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.conact_size,)),
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=ActivationFunctions[self.config.activation_action],
            device=self.device
        )

        q_network = HybridActionValueCritic(
            representation=deepcopy(representation),
            action_space=self.action_space,
            critic_hidden_size=self.config.qnetwork_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )

        model = MultipassParameterizedDQN(
            continuous_actor=continuous_actor,
            q_network=q_network,
            conact_sizes=self.conact_sizes
        )

        return model

