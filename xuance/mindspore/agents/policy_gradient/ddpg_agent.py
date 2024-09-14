import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecEnv
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import OffPolicyAgent


class DDPG_Agent(OffPolicyAgent):
    """The implementation of DDPG agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DDPG_Agent, self).__init__(config, envs)
        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy
        if self.config.policy == "DDPG_Policy":
            policy = REGISTRY_Policy["DDPG_Policy"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer,
                activation=activation, activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"DDPG currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self, observations: np.ndarray, test_mode: Optional[bool] = False):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
        """
        _, actions_output = self.policy(observations)
        if test_mode:
            actions = actions_output.numpy()
        else:
            actions = self.exploration(actions_output.numpy())
        return {"actions": actions}
