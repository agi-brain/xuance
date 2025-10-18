import gymnasium as gym
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gymnasium import spaces
from xuance.common import Optional, DummyOffPolicyBuffer, BaseCallback
from xuance.environment.single_agent_env import Gym_Env
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, InitializeFunctions, ActivationFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents import Agent
from xuance.mindspore.agents.policy_gradient.pdqn_agent import PDQN_Agent


class SPDQN_Agent(PDQN_Agent, Agent):
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
        Agent.__init__(self, config, envs, callback)
        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        self.observation_space = envs.observation_space.spaces[0]
        old_as = envs.action_space
        num_disact = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((old_as.spaces[0],
                                              *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                               old_as.spaces[1].spaces[i].high, dtype=np.float32)
                                                for i in range(0, num_disact))))
        self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
        self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
        self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in
                             range(1, num_disact + 1)]
        self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
        self.auxiliary_info_shape = {}
        self.nenvs = 1
        self.epsilon = 1.0
        self.epsilon_steps = 1000
        self.epsilon_initial = 1.0
        self.epsilon_final = 0.1
        self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

        # Build policy, optimizer, scheduler.
        self.policy = self._build_policy()

        self.memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.buffer_action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=config.buffer_size,
                                           batch_size=config.batch_size)
        self.learner = self._build_learner(self.config, self.policy, self.callback)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "SPDQN_Policy":
            policy = REGISTRY_Policy["SPDQN_Policy"](
                observation_space=self.observation_space, action_space=self.action_space,
                representation=representation,
                conactor_hidden_size=self.config.conactor_hidden_size,
                qnetwork_hidden_size=self.config.qnetwork_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def train(self, train_steps=10000):
        train_info = {}
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        obs, _ = self.envs.reset()
        for _ in tqdm(range(train_steps)):
            step_info = {}
            disaction, conaction, con_actions = self.action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            if self.render: self.envs.render("human")
            acts = np.concatenate(([disaction], con_actions), axis=0).ravel()

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs, next_obs=next_obs, rewards=rewards, terminals=terminal,
                                        action=action, acts=acts, steps=steps,
                                        disaction=disaction, conaction=conaction, con_actions=con_actions,
                                        train_steps=train_steps)

            self.memory.store(obs, acts, rewards, terminal, next_obs)
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)

            scores += rewards
            obs = deepcopy(next_obs)

            if terminal:
                episode_info = {"returns-step": scores}
                scores = 0
                returns = 0
                episodes += 1
                self.end_episode(episodes)
                obs, _ = self.envs.reset()
                self.log_infos(step_info, self.current_step)
                train_info.update(episode_info)
                self.callback.on_train_episode_info(envs=self.envs, policy=self.policy,
                                                    use_wandb=self.use_wandb,
                                                    current_step=self.current_step,
                                                    current_episode=self.current_episode,
                                                    train_steps=train_steps)

            self.current_step += self.n_envs

            if self.noise_scale >= self.end_noise:
                self.noise_scale -= self.delta_noise

            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)

        return train_info
