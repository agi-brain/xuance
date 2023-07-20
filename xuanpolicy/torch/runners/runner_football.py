import copy
import os
import socket
from pathlib import Path
from .runner_sc2 import SC2_Runner, make_envs
from xuanpolicy.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class Football_Runner(SC2_Runner):
    def __init__(self, args):
        self.num_agents, self.num_adversaries = 0, 0
        super(Football_Runner, self).__init__(args)
        self.episode_buffer = {
            'obs': np.zeros((self.n_envs, self.num_agents, self.episode_length + 1, self.dim_obs), dtype=np.float32),
            'actions': np.zeros((self.n_envs, self.num_agents, self.episode_length,), dtype=np.float32),
            'state': np.zeros((self.n_envs, self.episode_length + 1, self.dim_state), dtype=np.float32),
            'rewards': np.zeros([self.n_envs, self.episode_length, self.num_agents], dtype=np.float32),
            'terminals': np.zeros([self.n_envs, self.episode_length, 1], dtype=np.bool),
            'avail_actions': np.ones([self.n_envs, self.num_agents, self.episode_length + 1, self.dim_act], dtype=np.bool),
            'filled': np.zeros((self.n_envs, self.episode_length, 1), dtype=np.bool),
        }

    def get_agent_num(self):
        self.num_agents, self.num_adversaries = self.envs.num_agents, self.envs.num_adversaries

    def store_data(self, t_envs, obs_n, actions_dict, state, rewards, terminated, avail_actions):
        self.episode_buffer['obs'][self.env_ptr, :, t_envs] = obs_n
        self.episode_buffer['actions'][self.env_ptr, :, t_envs] = actions_dict['actions_n']
        self.episode_buffer['state'][self.env_ptr, t_envs] = state
        self.episode_buffer['rewards'][self.env_ptr, t_envs] = rewards
        self.episode_buffer['terminals'][self.env_ptr, t_envs] = terminated
        self.episode_buffer['avail_actions'][self.env_ptr, :, t_envs] = avail_actions

    def train_episode(self, n_episodes):
        step_info, episode_info, train_info = {}, {}, {}
        obs_n, state = self.envs.buf_obs, self.envs.buf_state
        rnn_hidden = self.rnn_hidden
        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        for _ in tqdm(range(n_episodes)):
            for step in range(self.episode_length):
                available_actions = self.envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, *rnn_hidden, test_mode=False)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
                self.filled[self.env_ptr, self.envs_step] = np.ones([self.n_envs, 1])
                rnn_hidden = actions_dict['rnn_hidden']
                self.store_data(self.envs_step, obs_n, actions_dict, state, rewards, terminated, available_actions)

                self.envs_step += 1
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:
                        rnn_hidden = self.agents.policy.representation.init_hidden_item(i_env, *rnn_hidden)
                        # store trajectory data:
                        last_avail_actions = info[i_env]["avail_actions"]
                        self.store_terminal_data(i_env, self.envs_step, obs_n, state, last_avail_actions, self.filled)
                        self.agents.memory.store(i_env, self.episode_buffer)
                        # prepare for next episode:
                        self.filled[i_env] = np.zeros([self.episode_length, 1], np.int32)
                        self.current_episode[i_env] += 1
                        self.envs_step[i_env] = 0
                        obs_n[i_env], state[i_env] = info[i_env]["reset_obs"], info[i_env]["reset_state"]
                        # Log episode info:
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i_env] = info[i_env]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i_env] = info[i_env]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i_env: info[i_env]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {"env-%d" % i_env: info[i_env]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

                self.current_step += self.n_envs

            incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
            incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
            win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
            episode_info["Train-Results/Win-Rate"] = win_rate

            train_info = self.agents.train(self.current_step)
            # Log train info:
            self.log_infos(train_info, self.current_step)
            self.log_infos(episode_info, self.current_step)
            self.rnn_hidden = rnn_hidden

    def test_episode(self, n_episodes):
        num_envs = self.test_envs.num_envs
        videos, episode_videos = [[] for _ in range(n_episodes)], []
        episode_score = []
        obs_n, state, infos = self.test_envs.reset()
        if self.args.render_mode == "rgb_array" and self.render:
            images = self.test_envs.render(self.args.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        best_score = -np.inf

        rnn_hidden = self.agents.policy.representation.init_hidden(num_envs)
        battles_game, battles_won = self.test_envs.battles_game.sum(), self.test_envs.battles_won.sum()
        for i_episode in range(n_episodes):
            for step in range(self.episode_length):
                available_actions = self.test_envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, *rnn_hidden, test_mode=True)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.test_envs.step(actions_dict['actions_n'])
                if self.args.render_mode == "rgb_array" and self.render:
                    images = self.test_envs.render(self.args.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)

                rnn_hidden = actions_dict['rnn_hidden']
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:
                        # prepare for next episode:
                        rnn_hidden = self.agents.policy.representation.init_hidden_item(i_env, *rnn_hidden)
                        obs_n[i_env], state[i_env] = info[i_env]["reset_obs"], info[i_env]["reset_state"]
                        episode_score.append(info[i_env]["episode_score"])
                        if best_score < episode_score[-1]:
                            best_score = episode_score[-1]
                            episode_videos = videos[i_env].copy()

        episode_score = np.array(episode_score)
        scores_mean = np.mean(episode_score)

        incre_battles_game = self.test_envs.battles_game.sum() - battles_game
        incre_battles_won = self.test_envs.battles_won.sum() - battles_won
        win_rate = float(incre_battles_won) / float(incre_battles_game) if incre_battles_game > 0 else 0.0

        if self.args.test_mode:
            print("Mean score: %.4f, Test Win Rate: %.4f." % (scores_mean, win_rate))

        # if self.args.render_mode == "rgb_array" and self.render:
        #     # time, height, width, channel -> time, channel, height, width
        #     videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
        #     self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        test_info = {
            "Test-Results/Mean-Episode-Rewards": scores_mean,
            "Test-Results/Win-Rate": win_rate,
        }
        self.log_infos(test_info, self.current_step)

        return episode_score, win_rate
