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
        self.fps = args.fps
        self.args = args
        self.render = args.render
        self.test_envs = None
        if args.logger == "tensorboard":
            time_string = time.asctime().replace(" ", "").replace(":", "_")
            log_dir = os.path.join(os.getcwd(), args.log_dir) + "/" + time_string
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif args.logger == "wandb":
            config_dict = vars(args)
            wandb_dir = Path(os.path.join(os.getcwd(), args.log_dir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=args.project_name,
                       entity=args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=args.env_id,
                       job_type=args.agent,
                       name=args.seed,
                       reinit=True
                       )
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.on_policy = self.args.on_policy
        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.envs_step = np.zeros((self.envs.num_envs,), np.int32)
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_length
        self.filled = np.zeros((self.n_envs, self.episode_length, 1), np.int32)
        self.get_agent_num()
        args.n_agents = self.num_agents
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape, args.act_shape = (self.dim_obs, ), ()
        args.rew_shape = args.done_shape = (1, )
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space
        self.episode_buffer = {
            'obs': np.zeros((self.n_envs, self.num_agents, self.episode_length + 1) + args.obs_shape, dtype=np.float32),
            'actions': np.zeros((self.n_envs, self.num_agents, self.episode_length) + args.act_shape, dtype=np.float32),
            'state': np.zeros((self.n_envs, self.episode_length + 1) + args.state_space.shape, dtype=np.float32),
            'rewards': np.zeros((self.n_envs, self.num_agents, self.episode_length) + args.rew_shape, dtype=np.float32),
            'terminals': np.zeros((self.n_envs, self.episode_length) + args.done_shape, dtype=np.bool),
            'avail_actions': np.ones((self.n_envs, self.num_agents, self.episode_length + 1, self.dim_act), dtype=np.bool),
            'filled': np.zeros((self.n_envs, self.episode_length, 1), dtype=np.bool),
        }
        if self.on_policy:
            self.episode_buffer.update({
                'values': np.zeros((self.n_envs, self.num_agents, self.episode_length) + args.rew_shape, np.float32),
                'returns': np.zeros((self.n_envs, self.num_agents, self.episode_length) + args.rew_shape, np.float32),
                'advantages': np.zeros((self.n_envs, self.num_agents, self.episode_length) + args.rew_shape, np.float32),
                'log_pi_old': np.zeros((self.n_envs, self.num_agents, self.episode_length,), np.float32)
            })
        self.env_ptr = range(self.n_envs)

        # environment details, representations, policies, optimizers, and agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)
        # initialize hidden units for RNN.
        self.rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs * self.num_agents)
        if self.on_policy:
            self.rnn_hidden_critic = self.agents.policy.representation_critic.init_hidden(self.n_envs * self.num_agents)
        else:
            self.rnn_hidden_critic = None

    def get_agent_num(self):
        self.num_agents, self.num_enemies = self.envs.num_agents, self.envs.num_enemies

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

    def get_actions(self, obs_n, avail_actions, *rnn_hidden, state=None, test_mode=False):
        log_pi_n, values_n, actions_n_onehot = None, None, None
        rnn_hidden_policy, rnn_hidden_critic = rnn_hidden[0], rnn_hidden[1]
        if self.on_policy:
            rnn_hidden_next, actions_n, log_pi_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                   avail_actions=avail_actions,
                                                                   test_mode=test_mode)
            if test_mode:
                rnn_hidden_critic_next, values_n = None, 0
            else:
                rnn_hidden_critic_next, values_n = self.agents.values(obs_n, *rnn_hidden_critic,
                                                                      state=state)
        else:
            rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                         avail_actions=avail_actions, test_mode=test_mode)
            rnn_hidden_critic_next = None
        return {'actions_n': actions_n, 'log_pi': log_pi_n,
                'rnn_hidden': rnn_hidden_next, 'rnn_hidden_critic': rnn_hidden_critic_next,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def store_data(self, t_envs, obs_n, actions_dict, state, rewards, terminated, avail_actions):
        self.episode_buffer['obs'][self.env_ptr, :, t_envs] = obs_n
        self.episode_buffer['actions'][self.env_ptr, :, t_envs] = actions_dict['actions_n']
        self.episode_buffer['state'][self.env_ptr, t_envs] = state
        self.episode_buffer['rewards'][self.env_ptr, :, t_envs] = rewards
        self.episode_buffer['terminals'][self.env_ptr, t_envs] = terminated
        self.episode_buffer['avail_actions'][self.env_ptr, :, t_envs] = avail_actions
        if self.on_policy:
            self.episode_buffer['values'][self.env_ptr, :, t_envs] = actions_dict['values']
            self.episode_buffer['log_pi_old'][self.env_ptr, :, t_envs] = actions_dict['log_pi']

    def store_terminal_data(self, i_env, t_env, obs_n, state, last_avail_actions, filled):
        self.episode_buffer['obs'][i_env, :, t_env] = obs_n[i_env]
        self.episode_buffer['state'][i_env, t_env] = state[i_env]
        self.episode_buffer['avail_actions'][i_env, :, t_env] = last_avail_actions
        self.episode_buffer['filled'][i_env] = filled[i_env]

    def train_episode(self, n_episodes):
        step_info, episode_info, train_info = {}, {}, {}
        obs_n, state = self.envs.buf_obs, self.envs.buf_state
        rnn_hidden, rnn_hidden_critic = self.rnn_hidden, self.rnn_hidden_critic
        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        dead_allies, dead_enemies = self.envs.dead_allies_count.sum(), self.envs.dead_enemies_count.sum()
        for _ in tqdm(range(n_episodes)):
            for step in range(self.episode_length):
                available_actions = self.envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                                state=state, test_mode=False)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
                self.filled[self.env_ptr, self.envs_step] = np.ones([self.n_envs, 1])
                self.store_data(self.envs_step, obs_n, actions_dict, state, rewards, terminated, available_actions)

                self.envs_step += 1
                rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:  # terminated
                        batch_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                        rnn_hidden = self.agents.policy.representation.init_hidden_item(batch_select,
                                                                                        *rnn_hidden)
                        # store trajectory data:
                        last_avail_actions = info[i_env]["avail_actions"]
                        self.store_terminal_data(i_env, self.envs_step, obs_n, state, last_avail_actions, self.filled)
                        if self.on_policy:
                            if terminated[i_env]:
                                values_next = np.array([0.0 for _ in range(self.num_agents)])
                            else:
                                rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(batch_select,
                                                                                                          *rnn_hidden_critic)
                                _, values_next = self.agents.values([obs_n[i_env]], *rnn_h_critic_i, state=[state[i_env]])
                            rnn_hidden_critic = self.agents.policy.representation_critic.init_hidden_item(batch_select,
                                                                                                          *rnn_hidden_critic)
                            self.agents.memory.finish_path(values_next, i_env, episode_data=self.episode_buffer,
                                                           current_t=self.envs_step[i_env],
                                                           value_normalizer=self.agents.learner.value_normalizer)
                            train_info = self.agents.train(self.current_step)

                            self.log_infos(train_info, self.current_step)
                        else:
                            self.agents.memory.store(self.episode_buffer, i_env)
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
            allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
            incre_allies = float(self.envs.dead_allies_count.sum() - dead_allies)
            incre_enemies = float(self.envs.dead_enemies_count.sum() - dead_enemies)
            dead_ratio = incre_allies / allies_count if allies_count > 0 else 0.0
            enemy_dead_ratio = incre_enemies / enemies_count if enemies_count > 0 else 0.0

            episode_info["Train-Results/Win-Rate"] = win_rate
            episode_info["Train-Results/Dead-Ratio"] = dead_ratio
            episode_info["Train-Results/Enemy-Dead-Ratio"] = enemy_dead_ratio
            self.log_infos(episode_info, self.current_step)

            if not self.on_policy:
                train_info = self.agents.train(self.current_step)
                self.log_infos(train_info, self.current_step)

        self.rnn_hidden, self.rnn_hidden_critic = rnn_hidden, rnn_hidden_critic

    def test_episode(self, n_episodes):
        num_envs = self.test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_score = []
        obs_n, state, infos = self.test_envs.reset()
        if self.args.render_mode == "rgb_array" and self.render:
            images = self.test_envs.render(self.args.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        best_score = -np.inf

        rnn_hidden = self.agents.policy.representation.init_hidden(num_envs * self.num_agents)

        battles_game, battles_won = self.test_envs.battles_game.sum(), self.test_envs.battles_won.sum()
        dead_allies, dead_enemies = self.test_envs.dead_allies_count.sum(), self.test_envs.dead_enemies_count.sum()
        for i_episode in range(n_episodes):
            for step in range(self.episode_length):
                available_actions = self.test_envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, None, test_mode=True)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.test_envs.step(actions_dict['actions_n'])
                if self.args.render_mode == "rgb_array" and self.render:
                    images = self.test_envs.render(self.args.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)

                rnn_hidden = actions_dict['rnn_hidden']
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)
                for i_env in range(num_envs):
                    if terminated[i_env] or truncated[i_env]:
                        # prepare for next episode:
                        agent_hidden_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                        rnn_hidden = self.agents.policy.representation.init_hidden_item(agent_hidden_select,
                                                                                        *rnn_hidden)
                        obs_n[i_env], state[i_env] = info[i_env]["reset_obs"], info[i_env]["reset_state"]
                        episode_score.append(info[i_env]["episode_score"])
                        if best_score < episode_score[-1]:
                            best_score = episode_score[-1]
                            episode_videos = videos[i_env].copy()
                        videos[i_env] = []

        episode_score = np.array(episode_score)
        scores_mean = np.mean(episode_score)

        incre_battles_game = self.test_envs.battles_game.sum() - battles_game
        incre_battles_won = self.test_envs.battles_won.sum() - battles_won
        win_rate = float(incre_battles_won) / float(incre_battles_game) if incre_battles_game > 0 else 0.0
        allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
        incre_allies = self.test_envs.dead_allies_count.sum() - dead_allies
        incre_enemies = self.test_envs.dead_enemies_count.sum() - dead_enemies
        dead_ratio = float(incre_allies) / allies_count if allies_count > 0 else 0.0
        enemy_dead_ratio = float(incre_enemies) / enemies_count if enemies_count > 0 else 0.0

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

        return episode_score, win_rate

    def run(self):
        if self.args.test_mode:
            arg_test = deepcopy(self.args)
            arg_test.parallels = 1
            self.render = arg_test.render = True
            self.test_envs = make_envs(arg_test)
            n_test_episodes = self.args.test_episode
            self.agents.load_model(self.agents.model_dir)
            self.test_episode(n_test_episodes)
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
        arg_test = deepcopy(self.args)
        arg_test.parallels = 1
        self.test_envs = make_envs(arg_test)

        n_train_episodes = self.args.running_steps // self.n_envs // self.episode_length
        n_eval_interval = self.args.eval_interval // self.n_envs // self.episode_length
        n_test_episodes = self.args.test_episode
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_episode_score, test_win_rate = self.test_episode(n_test_episodes)
        best_score = {
            "mean": test_episode_score.mean(),
            "std": test_episode_score.std(),
            "step": self.current_step
        }
        best_win_rate = test_win_rate

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.train_episode(n_episodes=n_eval_interval)
            test_episode_score, test_win_rate = self.test_episode(n_test_episodes)

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
        print("Best Score: ", best_score["mean"], "Std: ", best_score["std"])
        print("Best Win Rate: {}%".format(best_win_rate * 100))

        self.envs.close()
        self.test_envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

