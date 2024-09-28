import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyAgent
from xuance.common import RecurrentOffPolicyBuffer, EpisodeBuffer


class DRQN_Agent(OffPolicyAgent):
    """The implementation of Deep Recurrent Q-Netowrk (DRQN) agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DRQN_Agent, self).__init__(config, envs)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = config.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.policy = self._build_policy()  # build policy
        self.auxiliary_info_shape = {}
        self.memory = self._build_memory(auxiliary_info_shape=self.auxiliary_info_shape)  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
        self.lstm = True if config.rnn == "LSTM" else False

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = True if self.config.env_name == "Atari" else False
        Buffer = RecurrentOffPolicyBuffer(self.observation_space,
                                          self.action_space,
                                          auxiliary_info_shape,
                                          self.n_envs,
                                          self.config.buffer_size,
                                          self.config.batch_size,
                                          episode_length=self.episode_length,
                                          lookup_length=self.config.lookup_length)
        return Buffer

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "DRQN_Policy":
            policy = REGISTRY_Policy["DRQN_Policy"](
                action_space=self.action_space, representation=representation,
                rnn=self.config.rnn, recurrent_hidden_size=self.config.recurrent_hidden_size,
                recurrent_layer_N=self.config.recurrent_layer_N, dropout=self.config.dropout,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(
                f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self, obs, egreedy=0.0, rnn_hidden=None):
        _, argmax_action, _, rnn_hidden_next = self.policy(obs[:, np.newaxis], *rnn_hidden)
        random_action = np.random.choice(self.action_space.n, self.n_envs)
        if np.random.rand() < egreedy:
            actions = random_action
        else:
            actions = argmax_action.detach().cpu().numpy()
        return {"actions": actions, "rnn_hidden_next": rnn_hidden_next}

    def train(self, train_steps):
        obs = self.envs.buf_obs
        episode_data = [EpisodeBuffer() for _ in range(self.n_envs)]
        for i_env in range(self.n_envs):
            episode_data[i_env].obs.append(self._process_observation(obs[i_env]))
        self.rnn_hidden = self.policy.init_hidden(self.n_envs)
        dones = [False for _ in range(self.n_envs)]
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, self.egreedy, self.rnn_hidden)
            acts, self.rnn_hidden = policy_out['actions'], policy_out['rnn_hidden_next']
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            if (self.current_step > self.start_training) and (self.current_step % self.training_frequency == 0):
                # training
                train_infos = self.train_epochs(n_epochs=1)
                self.log_infos(train_infos, self.current_step)

            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                episode_data[i].put(
                    [self._process_observation(obs[i]), acts[i], self._process_reward(rewards[i]), terminals[i]])
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        self.rnn_hidden = self.policy.init_hidden_item(self.rnn_hidden, i)
                        dones[i] = True
                        self.current_episode[i] += 1
                        if self.rank == 0:
                            if self.use_wandb:
                                step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = infos[i]["episode_step"]
                                step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                            else:
                                step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                                step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                                    f"env-{i}": infos[i]["episode_score"]}
                            self.log_infos(step_info, self.current_step)
                        self.memory.store(episode_data[i])
                        episode_data[i] = EpisodeBuffer()
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        episode_data[i].obs.append(self._process_observation(obs[i]))

            self.current_step += self.n_envs
            if self.egreedy > self.end_greedy:
                self.egreedy = self.egreedy - self.delta_egreedy

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        rnn_hidden = self.policy.init_hidden(num_envs)
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, egreedy=0.0, rnn_hidden=rnn_hidden)
            acts, rnn_hidden = policy_out['actions'], policy_out['rnn_hidden_next']
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        rnn_hidden = self.policy.init_hidden_item(rnn_hidden, i)
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
