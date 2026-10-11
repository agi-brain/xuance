import gymnasium
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from typing import Optional, Tuple
from xuance.common import BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.mindspore import ms, ops, Tensor, Module
from xuance.mindspore.utils import ActivationFunctions
from xuance.mindspore.agents import OnPolicyAgent
from xuance.mindspore.rl_models.heads import GaussianActorHead, CategoricalActorHead, ValueHead
from xuance.mindspore.rl_models import CategoricalActor, GaussianActor
from xuance.mindspore.rl_models import StateValueCritic as Critic
from xuance.mindspore.rl_models import ActorCritic, SharedActorCritic
from xuance.mindspore.rl_models.modules import ActionOutput


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
        actor_input = dict(normalizer=self.normalize_fn,
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
                               normalizer=self.normalize_fn,
                               initializer=self.initializer,
                               activation=self.activation)
            model = SharedActorCritic(representation=representation, actor=actor, critic=critic)
        else:
            critic = Critic(representation=deepcopy(representation),
                            critic_hidden_size=self.config.critic_hidden_size,
                            normalizer=self.normalize_fn,
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
        observations = Tensor(observations, dtype=ms.float32)
        values_next = self._values_step(observations)
        values_next = values_next.asnumpy()
        return values_next

    @ms.jit
    def _values_step(self, observations: Tensor, **kwargs) -> Tensor:
        value = self.model.values(observations)
        return value

    @ms.jit
    def _rollout_step(self, observations: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        if self.model.is_continuous:
            a_mean, log_std, values = self.model(observations)
            a_std = ops.exp(log_std)
            action_sampled = a_mean + a_std * ops.randn_like(a_mean)
            # calculate log prob
            action_log_probs = -0.5 * (
                    ((action_sampled - a_mean) / (a_std + 1e-8)) ** 2 + 2.0 * log_std + ops.log(Tensor(2.0 * np.pi)))
            action_log_probs = action_log_probs.squeeze(-1)
        else:
            logits, values = self.model(observations)
            pi_dists = ops.RandomCategorical(ms.int32)
            action_sampled = pi_dists(logits, 1, 0).squeeze(-1)
            log_probs = ops.log_softmax(logits, axis=-1)
            action_log_probs = ops.gather_elements(log_probs, -1, action_sampled.expand_dims(-1)).squeeze(-1)
        return action_sampled, action_log_probs, values

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
        observations = Tensor(observations, dtype=ms.float32)

        actions, log_pi, values = self._rollout_step(observations)

        if not self.is_tensor_memory:
            actions = actions.numpy()
            log_pi = log_pi.numpy()
            values = values.numpy()

        return ActionOutput(
            env_actions=actions,
            values=values,
            log_probs=log_pi
        )

    def train(self, train_steps):
        train_info = {}
        obs = self.train_envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.get_actions(obs, return_dists=False, return_logpi=True)
            acts = policy_out.env_actions
            value = policy_out.values
            next_obs, rewards, terminals, truncations, infos = self.train_envs.step(acts)
            aux_info = self.get_aux_info(policy_out)

            self.callback.on_train_step(self.current_step, envs=self.train_envs, model=self.model,
                                        obs=obs, policy_out=policy_out, acts=acts, vals=value, next_obs=next_obs,
                                        rewards=rewards, terminals=terminals, truncations=truncations,
                                        infos=infos, aux_info=aux_info, train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, aux_info)
            if self.memory.full:
                vals = self.get_terminated_values(next_obs)
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                update_info = self.train_epochs(self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, model=self.model, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (not truncations[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0.0, i)
                        else:
                            vals = self.get_terminated_values(next_obs)
                            self.memory.finish_path(vals[i], i)
                        obs[i] = infos[i]["reset_obs"]
                        self.train_envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.train_envs, model=self.model, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)
            self.current_step += self.n_envs
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                            train_steps=train_steps, train_info=train_info)
        return train_info
