import numpy as np
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.rl_models import DeterministicActor, ActionValueCritic
from xuance.torch.rl_models.architectures import DeterministicActorCritic


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
            activation_action=ActivationFunctions[self.config.activation_action],
            device=self.device
        )

        # build critic network
        critic = ActionValueCritic(
            representation=deepcopy(representation),
            action_space=self.action_space,
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )

        # build the RL model
        model = DeterministicActorCritic(actor=actor, critic=critic)

        return model

    def get_actions(self, observations: np.ndarray,
                    test_mode: Optional[bool] = False):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
        """
        actions_output = self.model.act(observations)
        if test_mode:
            actions = actions_output.detach().cpu().numpy()
        else:
            actions = self.exploration(actions_output.detach().cpu().numpy())
        return {"actions": actions}
