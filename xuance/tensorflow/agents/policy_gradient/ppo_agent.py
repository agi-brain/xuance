import gymnasium
from copy import deepcopy
from argparse import Namespace

import numpy as np
from gymnasium.spaces import Space
from typing import Optional, Tuple
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow import tf, Tensor, Module
from xuance.tensorflow.utils import ActivationFunctions
from xuance.tensorflow.agents import OnPolicyAgent
from xuance.tensorflow.rl_models.heads import GaussianActorHead, CategoricalActorHead, ValueHead
from xuance.tensorflow.rl_models import CategoricalActor, GaussianActor
from xuance.tensorflow.rl_models import StateValueCritic as Critic
from xuance.tensorflow.rl_models import ActorCritic, SharedActorCritic
from xuance.tensorflow.rl_models.modules import ActionOutput


class PPO_Agent(OnPolicyAgent):
    """The implementation of PPO agent.

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
        super(PPO_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory(self.auxiliary_info_shape)  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        shared_representation = getattr(self.config, "shared_representation", True)

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network
        actor_input = dict(normalizer=self.normalizer_fn,
                           initializer=self.initializer,
                           activation=self.activation)
        if shared_representation:
            actor_input.update(dict(feature_dim=representation.output_shapes['state'][0],
                                    hidden_size=self.config.actor_hidden_size))
            if isinstance(self.action_space, gymnasium.spaces.Box):
                actor_input.update(dict(action_dim=self.action_space.shape[0],
                                        activation_action=ActivationFunctions[self.config.activation_action], ))
                actor = GaussianActorHead(**actor_input)
            elif isinstance(self.action_space, gymnasium.spaces.Discrete):
                actor_input.update(dict(action_dim=self.action_space.n))
                actor = CategoricalActorHead(**actor_input)
            else:
                raise NotImplementedError
        else:
            actor_input.update(dict(representation=representation,
                                    actor_hidden_size=self.config.actor_hidden_size,
                                    action_space=self.action_space))
            if isinstance(self.action_space, gymnasium.spaces.Box):
                actor_input.update(dict(activation_action=ActivationFunctions[self.config.activation_action], ))
                actor = GaussianActor(**actor_input)
            elif isinstance(self.action_space, gymnasium.spaces.Discrete):
                actor = CategoricalActor(**actor_input)
            else:
                raise NotImplementedError

        # build critic network and the RL model
        if shared_representation:
            critic = ValueHead(feature_dim=representation.output_shapes['state'][0],
                               hidden_size=self.config.critic_hidden_size,
                               normalizer=self.normalizer_fn,
                               initializer=self.initializer,
                               activation=self.activation)
            model = SharedActorCritic(representation=representation, actor=actor, critic=critic)
        else:
            critic = Critic(representation=deepcopy(representation),
                            critic_hidden_size=self.config.critic_hidden_size,
                            normalizer=self.normalizer_fn,
                            initializer=self.initializer,
                            activation=self.activation)
            model = ActorCritic(actor=actor, critic=critic)

        return model

    @property
    def auxiliary_info_shape(self):
        return {"old_logp": ()}

    def get_aux_info(self, policy_output: ActionOutput = None):
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        aux_info = {"old_logp": policy_output.log_probs}
        return aux_info

    def get_terminated_values(self, observations_next: np.ndarray, rewards: np.ndarray = None) -> np.ndarray:
        """Compute value estimates for terminal/terminated states.

        This method evaluates the value function on terminal observations and returns the value estimates used for
        bootstrapping (e.g., when finishing a trajectory segment).

        Args:
            observations_next (np.ndarray): Observations at the terminal step
                (or the next observations used for bootstrapping).
            rewards (Optional[np.ndarray]): Rewards corresponding to the terminal transitions.
                This argument is reserved for algorithm-specific implementations and may be unused.

        Returns:
            np.ndarray: Value estimates for the provided terminal observations.
        """
        observations = self._process_observation(observations_next)
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        values_next = self._values_step(observations)
        values_next = values_next.numpy()
        return values_next

    @tf.function
    def _values_step(self, observations: Tensor, **kwargs) -> Tensor:
        value = self.model.values(observations)
        return value

    @tf.function
    def _stochastic_rollout_step(self, observations: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        model_output = self.model(observations)
        policy_dists = model_output.distributions
        actions = policy_dists.stochastic_sample()
        log_pi = policy_dists.log_prob(actions)
        values = model_output.values
        return actions, log_pi, values

    @tf.function
    def _deterministic_rollout_step(self, observations: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        model_output = self.model(observations)
        policy_dists = model_output.distributions
        actions = policy_dists.deterministic_sample()
        log_pi = policy_dists.log_prob(actions)
        values = model_output.values
        return actions, log_pi, values

    def get_actions(
            self,
            observations: np.ndarray,
            deterministic: bool = False,
            **kwargs
    ) -> ActionOutput:
        """Compute actions and value estimates for a batch of observations.

        This method performs a forward pass through the current policy to obtain action distributions
        and value predictions. Actions are sampled stochastically from the policy distribution.

        Args:
            observations (np.ndarray): Batch of observations. The array is expected to have shape compatible with
                the underlying policy.
            deterministic (bool): True for deterministic policy and False for stochastic policy.

        Returns:
            ActionOutput.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)

        if deterministic:
            actions, log_pi, values = self._deterministic_rollout_step(observations)
        else:
            actions, log_pi, values = self._stochastic_rollout_step(observations)

        if self.is_tensor_memory:
            values = 0 if values is None else values
        else:
            actions = actions.numpy()
            log_pi = log_pi.numpy()
            values = 0 if values is None else values.numpy()

        return ActionOutput(
            env_actions=actions,
            values=values,
            log_probs=log_pi
        )
