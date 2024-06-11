import gym
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gym import spaces
from xuance.environment.single_agent_env import Gym_Env
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import PDQN_Learner
from xuance.torch.agents import Agent
from xuance.common import DummyOffPolicyBuffer


class PDQN_Agent(Agent):
    """The implementation of PDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Gym_Env):
        super(PDQN_Agent, self).__init__(config, envs)

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

        # Build policy, optimizer, scheduler.
        self.policy = self._build_policy()
        conactor_optimizer = torch.optim.Adam(self.policy.conactor.parameters(), self.config.learning_rate)
        qnetwork_optimizer = torch.optim.Adam(self.policy.qnetwork.parameters(), self.config.learning_rate)
        optimizers = [conactor_optimizer, qnetwork_optimizer]
        conactor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(conactor_optimizer, start_factor=1.0, end_factor=0.25,
                                                                  total_iters=self.config.running_steps)
        qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(qnetwork_optimizer, start_factor=1.0, end_factor=0.25,
                                                                  total_iters=self.config.running_steps)
        lr_schedulers = [conactor_lr_scheduler, qnetwork_lr_scheduler]

        self.memory = DummyOffPolicyBuffer(observation_space=self.observation_space,
                                           action_space=self.buffer_action_space,
                                           auxiliary_shape=self.auxiliary_info_shape,
                                           n_envs=self.n_envs,
                                           buffer_size=config.buffer_size,
                                           batch_size=config.batch_size)
        self.learner = self._build_learner(self.config, envs.max_episode_steps, self.policy, optimizers, lr_schedulers)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact + 1)])
        self.conact_size = int(self.conact_sizes.sum())

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.config)

        # build policy.
        if self.config.policy == "PDQN_Policy":
            policy = REGISTRY_Policy["PDQN_Policy"](
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

    def _build_learner(self, *args):
        return PDQN_Learner(*args)

    def action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            con_actions = self.policy.con_action(obs)
            rnd = np.random.rand()
            if rnd < self.epsilon:
                disaction = np.random.choice(self.num_disact)
            else:
                q = self.policy.Qeval(obs.unsqueeze(0), con_actions.unsqueeze(0))
                q = q.detach().cpu().data.numpy()
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
            self.memory.store(obs, acts, rewards, terminal, next_obs)
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=1)
                self.log_infos(train_info, self.current_step)

            scores += rewards
            obs = deepcopy(next_obs)

            if terminal:
                step_info["returns-step"] = scores
                scores = 0
                returns = 0
                episodes += 1
                self.end_episode(episodes)
                obs, _ = self.envs.reset()
                self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs

            if self.egreedy >= self.end_greedy:
                self.egreedy -= self.delta_egreedy

            if self.noise_scale >= self.end_noise:
                self.noise_scale -= self.delta_noise

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        episode_score = 0
        current_episode, scores, best_score = 0, [], -np.inf
        obs, _ = self.envs.reset()

        while current_episode < test_episodes:
            disaction, conaction, con_actions = self.action(obs)
            action = self.pad_action(disaction, conaction)
            action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                disaction]
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            self.envs.render("human")
            episode_score += rewards
            obs = deepcopy(next_obs)
            if terminal:
                scores.append(episode_score)
                obs, _ = self.envs.reset()
                current_episode += 1
                if best_score < episode_score:
                    best_score = episode_score
                episode_score = 0
                if self.config.test_mode:
                    print("Episode: %d, Score: %.2f" % (current_episode, episode_score))

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores

    def end_episode(self, episode):
        if episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
