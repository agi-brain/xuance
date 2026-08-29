import numpy as np
from argparse import Namespace
from gymnasium import spaces
from xuance.common import Optional, BaseCallback
from xuance.environment.single_agent_env import Gym_Env
from xuance.tensorflow import Module, ModuleList
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents.policy_gradient.pdqn_agent import PDQN_Agent
from xuance.tensorflow.rl_models import DeterministicActor, ActionValueCritic
from xuance.tensorflow.rl_models.architectures import SplitParameterisedDQN


class SPDQN_Agent(PDQN_Agent):
    """The implementation of SPDQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 callback: Optional[BaseCallback] = None):
        super(SPDQN_Agent, self).__init__(config, envs, callback=callback)

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build the RL model.
        continuous_actor = DeterministicActor(
            representation=representation,
            actor_hidden_size=self.config.conactor_hidden_size,
            action_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.conact_size,)),
            normalizer=self.normalizer_fn,
            initializer=self.initializer,
            activation=self.activation,
            activation_action=ActivationFunctions[self.config.activation_action]
        )

        q_network = ModuleList()
        for k in range(self.num_disact):
            q_network.append(ActionValueCritic(
                representation=representation.clone(copy_weights=False, trainable=True,
                                                    name=f"critic_representation_{k}"),
                action_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.conact_sizes[k],)),
                critic_hidden_size=self.config.qnetwork_hidden_size,
                normalizer=self.normalizer_fn,
                initializer=self.initializer,
                activation=self.activation
            ))

        model = SplitParameterisedDQN(
            continuous_actor=continuous_actor,
            q_network=q_network,
            conact_sizes=self.conact_sizes,
            num_disact=self.action_space.spaces[0].n
        )

        return model
