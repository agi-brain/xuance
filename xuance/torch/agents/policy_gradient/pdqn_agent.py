import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gymnasium import spaces
from xuance.common import Optional, DummyOffPolicyBuffer, BaseCallback
from xuance.environment.single_agent_env import Gym_Env
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import Agent
from xuance.torch.rl_models import DeterministicActor, HybridActionValueCritic
from xuance.torch.rl_models.architectures import ParameterizedDQN


class PDQN_Agent(Agent):
    """The implementation of PDQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env,
                 callback: Optional[BaseCallback] = None):
        super(PDQN_Agent, self).__init__(config, envs, callback)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

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

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

        # Build RL model, optimizer, scheduler.
        self.model = self._build_model()

        self.memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.buffer_action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=config.buffer_size,
                                           batch_size=config.batch_size)
        self.learner = self._build_learner(self.config, self.model, self.callback)

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

        model = ParameterizedDQN(
            continuous_actor=continuous_actor,
            q_network=q_network
        )

        return model

    @torch.no_grad()
    def get_actions(self, obs):
        obs = torch.as_tensor(obs, device=self.device).float()
        con_actions = self.model.con_action(obs)
        rnd = np.random.rand()
        if rnd < self.epsilon:
            disaction = np.random.choice(self.num_disact)
        else:
            q = self.model.Qeval(obs.unsqueeze(0), con_actions.unsqueeze(0))
            q = q.cpu().data.numpy()
            disaction = np.argmax(q)

        con_actions = con_actions.cpu().data.numpy()
        offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
        conaction = con_actions[offset:offset + self.conact_sizes[disaction]]

        return disaction, conaction, con_actions

    def pad_action(self, disaction, conaction):
        con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32),
                       np.zeros((1,), dtype=np.float32)]
        con_actions[disaction][:] = conaction
        return (disaction, con_actions)

    def train_epochs(self, n_epochs=1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample()
            train_info = self.learner.update(**samples)
        return train_info

    def train(self, train_steps=10000):
        train_info = {}
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        obs, _ = self.train_envs.reset()
        for _ in tqdm(range(train_steps)):
            disaction, conaction, con_actions = self.get_actions(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = self.train_envs.step(action)
            if self.render: self.train_envs.render("human")
            acts = np.concatenate(([disaction], con_actions), axis=0).ravel()

            self.callback.on_train_step(self.current_step, envs=self.train_envs, model=self.model,
                                        obs=obs, next_obs=next_obs, rewards=rewards, terminals=terminal,
                                        action=action, acts=acts, steps=steps,
                                        disaction=disaction, conaction=conaction, con_actions=con_actions,
                                        train_steps=train_steps)

            self.memory.store(obs, acts, rewards, terminal, next_obs)
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, model=self.model, memory=self.memory,
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
                obs, _ = self.train_envs.reset()
                self.log_infos(episode_info, self.current_step)
                train_info.update(episode_info)
                self.callback.on_train_episode_info(envs=self.train_envs, model=self.model,
                                                    rank=self.rank, use_wandb=self.use_wandb,
                                                    current_step=self.current_step,
                                                    current_episode=self.current_episode,
                                                    train_steps=train_steps)

            self.current_step += self.n_envs

            if self.egreedy >= self.end_greedy:
                self.egreedy -= self.delta_egreedy

            if self.noise_scale >= self.end_noise:
                self.noise_scale -= self.delta_noise

            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                            train_steps=train_steps, train_info=train_info)

        return train_info

    def test(self,
             test_episodes: int,
             test_envs: Gym_Env = None,
             close_envs: bool = True) -> list:
        if test_envs is None:
            raise ValueError("`test_envs` must be provided for evaluation.")
        episode_score = 0
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs, _ = test_envs.reset()

        while current_episode < test_episodes:
            disaction, conaction, con_actions = self.get_actions(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = test_envs.step(action)
            if self.config.render:
                test_envs.render("human")

            self.callback.on_test_step(envs=test_envs, model=self.model,
                                       disaction=disaction, conaction=conaction, con_actions=con_actions,
                                       action=action, steps=steps, rewards=rewards, terminals=terminal,
                                       obs=obs, next_obs=next_obs,
                                       current_train_step=self.current_step,
                                       current_step=current_step, current_episode=current_episode)

            episode_score += rewards
            obs = deepcopy(next_obs)
            if terminal:
                scores.append(episode_score)
                obs, _ = test_envs.reset()
                current_episode += 1
                if best_score < episode_score:
                    best_score = episode_score
                episode_score = 0

            current_step += 1

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        self.callback.on_test_end(envs=test_envs, model=self.model,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        test_envs.close()

        return scores

    def end_episode(self, episode):
        if episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
