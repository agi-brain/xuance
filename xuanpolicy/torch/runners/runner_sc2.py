import copy
import os
import socket
from pathlib import Path
from .runner_basic import Runner_Base, make_envs
from xuanpolicy.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class SC2_Runner(Runner_Base):
    def __init__(self, args):
        super(SC2_Runner, self).__init__(args)
        self.fps = 20
        self.args = args
        self.render = args.render
        if args.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), args.logdir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.logdir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=time.asctime(),
                       reinit=True
                       )
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.train_per_step = args.train_per_step
        self.current_step = 0
        self.envs_step = np.zeros((self.envs.num_envs,), np.int32)
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_length
        self.filled = np.zeros((self.n_envs, self.episode_length, 1), np.int32)
        self.rnn_hidden = None
        self.num_agents = args.n_agents = self.envs.num_agents
        self.num_enemies = self.envs.num_enemies
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape = (self.num_agents, self.dim_obs)
        args.act_shape = (self.num_agents, )
        args.rew_shape, args.done_shape = (1, ), (1, )
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space
        self.episode_buffer = {
            'obs': np.zeros((self.n_envs, self.num_agents, self.episode_length + 1, self.dim_obs), dtype=np.float32),
            'actions': np.zeros((self.n_envs, self.num_agents, self.episode_length, ), dtype=np.float32),
            'state': np.zeros((self.n_envs, self.episode_length + 1, self.dim_state), dtype=np.float32),
            'rewards': np.zeros([self.n_envs, self.episode_length, 1], dtype=np.float32),
            'terminals': np.zeros([self.n_envs, self.episode_length, 1], dtype=np.bool),
            'avail_actions': np.ones([self.n_envs, self.num_agents, self.episode_length + 1, self.dim_act], dtype=np.bool),
            'filled': np.zeros((self.n_envs, self.episode_length, 1), dtype=np.bool),
        }
        self.env_ptr = range(self.n_envs)

        # environment details, representations, policies, optimizers, and agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)
        self.rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs)
        self.last_battles_won, self.last_battles_game = 0, 0

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def get_actions(self, obs_n, avail_actions, *rnn_hidden, test_mode=False):
        log_pi_n, values_n, actions_n_onehot = [], [], []
        rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden,
                                                     avail_actions=avail_actions, test_mode=test_mode)
        # rnn_hidden_next = self.rnn_hidden
        return {'actions_n': actions_n, 'log_pi': log_pi_n,
                'rnn_hidden': rnn_hidden_next,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def store_data(self, t_envs, obs_n, actions_dict, state, rewards, terminated, avail_actions):
        self.episode_buffer['obs'][self.env_ptr, :, t_envs] = obs_n
        self.episode_buffer['actions'][self.env_ptr, :, t_envs] = actions_dict['actions_n']
        self.episode_buffer['state'][self.env_ptr, t_envs] = state
        self.episode_buffer['rewards'][self.env_ptr, t_envs] = rewards
        self.episode_buffer['terminals'][self.env_ptr, t_envs] = terminated
        self.episode_buffer['avail_actions'][self.env_ptr, :, t_envs] = avail_actions

    def store_terminal_data(self, i_env, t_env, obs_n, state, last_avail_actions, filled):
        self.episode_buffer['obs'][i_env, :, t_env] = obs_n[i_env]
        self.episode_buffer['state'][i_env, t_env] = state[i_env]
        self.episode_buffer['avail_actions'][i_env, :, t_env] = last_avail_actions
        self.episode_buffer['filled'][i_env] = filled[i_env]

    def train_episode(self, n_episodes):
        step_info, episode_info, train_info = {}, {}, {}
        obs_n, state = self.envs.buf_obs, self.envs.buf_state
        rnn_hidden = self.rnn_hidden
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

            battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
            incre_battles_game = float(battles_game - self.last_battles_game)
            win_rate = float(battles_won - self.last_battles_won) / incre_battles_game if incre_battles_game > 0 else 0.0
            all_allies_count = float(self.envs.battles_game.sum() * self.num_agents)
            all_enemies_count = float(self.envs.battles_game.sum() * self.num_enemies)
            dead_ratio = float(self.envs.dead_allies_count.sum()) / all_allies_count
            enemy_dead_ratio = float(self.envs.dead_enemies_count.sum()) / all_enemies_count
            episode_info["Train-Results/Win-Rate"] = win_rate
            episode_info["Train-Results/Dead-Ratio"] = dead_ratio
            episode_info["Train-Results/Enemy-Dead-Ratio"] = enemy_dead_ratio
            if not self.train_per_step:
                train_info = self.agents.train(self.current_step)
                # Log train info:
                self.log_infos(train_info, self.current_step)
                self.log_infos(episode_info, self.current_step)
            self.last_battles_game, self.last_battles_won = battles_game, battles_won
            self.rnn_hidden = rnn_hidden

    def test_episode(self, env_fn, n_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(n_episodes)], []
        episdoe_score = np.zeros([n_episodes, 1], dtype=np.float32)
        obs_n, state, infos = test_envs.reset()
        if self.args.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.args.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        best_score = -np.inf

        for i_episode in range(n_episodes):
            rnn_hidden = self.agents.policy.representation.init_hidden(num_envs)
            done = False
            while not done:
                available_actions = test_envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, *rnn_hidden, test_mode=True)
                next_obs_n, next_state, rewards, terminated, truncated, info = test_envs.step(actions_dict['actions_n'])
                if self.args.render_mode == "rgb_array" and self.render:
                    images = test_envs.render(self.args.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)

                rnn_hidden = actions_dict['rnn_hidden']
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                if (terminated[0] or truncated[0]) and not done:
                    # prepare for next episode:
                    done = True
                    obs_n[0], state[0] = info[0]["reset_obs"], info[0]["reset_state"]
                    if best_score < info[0]["episode_score"]:
                        best_score = info[0]["episode_score"]
                        episode_videos = videos[0].copy()
                    episdoe_score[i_episode] = info[0]["episode_score"]
        scores_mean = episdoe_score.mean()
        win_rate = float(test_envs.battles_won) / float(test_envs.battles_game)
        all_allies_count = float(test_envs.battles_game.sum() * self.num_agents)
        all_enemies_count = float(test_envs.battles_game.sum() * self.num_enemies)
        dead_ratio = float(test_envs.dead_allies_count.sum()) / all_allies_count
        enemy_dead_ratio = float(test_envs.dead_enemies_count.sum()) / all_enemies_count

        if self.args.test_mode:
            print("Mean score: %.4f, Test Win Rate: %.4f." % (scores_mean, win_rate))

        if self.args.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        test_info = {
            "Test-Results/Mean-Episode-Rewards": scores_mean,
            "Test-Results/Win-Rate": win_rate,
            "Test-Results/Dead-Ratio": dead_ratio,
            "Test-Results/Enemy-Dead-Ratio": enemy_dead_ratio,
        }
        self.log_infos(test_info, self.current_step)
        test_envs.close()

        return episdoe_score, win_rate

    def run(self):
        if self.args.test_mode:
            def env_fn():
                arg_test = copy.deepcopy(self.args)
                arg_test.parallels = 1
                return make_envs(arg_test)
            self.render = True
            n_test_episodes = self.args.test_episode
            self.agents.load_model(self.agents.modeldir)
            self.test_episode(env_fn, n_test_episodes)
            print("Finish testing.")
        else:
            n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
            self.train_episode(n_train_episodes)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def benchmark(self):
        def env_fn():
            arg_test = copy.deepcopy(self.args)
            arg_test.parallels = 1
            return make_envs(arg_test)

        n_train_episodes = self.args.running_steps // self.n_envs // self.episode_length
        n_eval_interval = self.args.eval_interval // self.n_envs // self.episode_length
        n_test_episodes = self.args.test_episode
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_episode_score, test_win_rate = self.test_episode(env_fn, n_test_episodes)
        best_score = {
            "mean": test_episode_score.mean(),
            "std": test_episode_score.std(),
            "step": self.current_step
        }
        best_win_rate = test_win_rate

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.train_episode(n_episodes=n_eval_interval)
            test_episode_score, test_win_rate = self.test_episode(env_fn, n_test_episodes)

            mean_test_scores = test_episode_score.mean()
            if best_score["mean"] < mean_test_scores:
                best_score = {
                    "mean": mean_test_scores,
                    "std": test_episode_score.std(),
                    "step": self.current_step
                }
            if best_win_rate < test_win_rate:
                best_win_rate = test_win_rate
            # save best model
            self.agents.save_model("best_model.pth")

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: ", best_score["mean"], "Std: ", best_win_rate["std"])
        print("Best Win Rate: ", best_win_rate)

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

