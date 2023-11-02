import os
import socket
from pathlib import Path
from .runner_basic import Runner_Base
from xuance.torch.agents import REGISTRY as REGISTRY_Agent
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from copy import deepcopy


class SC2_Runner(Runner_Base):
    def __init__(self, args):
        super(SC2_Runner, self).__init__(args)
        self.fps = args.fps
        self.args = args
        self.render = args.render
        self.test_envs = None

        time_string = time.asctime().replace(" ", "").replace(":", "_")
        seed = f"seed_{self.args.seed}_"
        self.args.model_dir_load = args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), args.model_dir, seed + time_string)
        if (not os.path.exists(self.args.model_dir_save)) and (not args.test_mode):
            os.makedirs(self.args.model_dir_save)

        if args.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), args.log_dir, seed + time_string)
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
                       reinit=True)
            self.use_wandb = True
        else:
            raise "No logger is implemented."

        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.env_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
        self.episode_length = self.envs.max_episode_length
        self.num_agents, self.num_enemies = self.get_agent_num()
        args.n_agents = self.num_agents
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape, args.act_shape = (self.dim_obs,), ()
        args.rew_shape = args.done_shape = (1,)
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space

        # environment details, representations, policies, optimizers, and agents.
        self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)
        self.on_policy = self.agents.on_policy

    def init_rnn_hidden(self):
        rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs * self.num_agents)
        if self.on_policy and self.args.agent != "COMA":
            rnn_hidden_critic = self.agents.policy.representation_critic.init_hidden(self.n_envs * self.num_agents)
        else:
            rnn_hidden_critic = [None, None]
        return rnn_hidden, rnn_hidden_critic

    def get_agent_num(self):
        return self.envs.num_agents, self.envs.num_enemies

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if x_index <= self.running_steps:
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
        if x_index <= self.running_steps:
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
            if self.args.agent == "COMA":
                rnn_hidden_next, actions_n, actions_n_onehot = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                               avail_actions=avail_actions,
                                                                               test_mode=test_mode)
            else:
                rnn_hidden_next, actions_n, log_pi_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                       avail_actions=avail_actions,
                                                                       test_mode=test_mode)
            if test_mode:
                rnn_hidden_critic_next, values_n = None, 0
            else:
                kwargs = {"state": state}
                if self.args.agent == "COMA":
                    kwargs.update({"actions_n": actions_n, "actions_onehot": actions_n_onehot})
                rnn_hidden_critic_next, values_n = self.agents.values(obs_n, *rnn_hidden_critic, **kwargs)
        else:
            rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                         avail_actions=avail_actions, test_mode=test_mode)
            rnn_hidden_critic_next = None
        return {'actions_n': actions_n, 'log_pi': log_pi_n,
                'rnn_hidden': rnn_hidden_next, 'rnn_hidden_critic': rnn_hidden_critic_next,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def run_episodes(self, n_episodes, test_mode=False):
        step_info, episode_info, train_info = {}, {}, {}
        videos, episode_videos = [[] for _ in range(self.n_envs)], []
        episode_score, best_score, win_rate = [], -np.inf, 0.0

        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        dead_allies, dead_enemies = self.envs.dead_allies_count.sum(), self.envs.dead_enemies_count.sum()
        for i_episode in range(n_episodes):
            # reset the envs and settings
            obs_n, state, info = self.envs.reset()
            envs_done = self.envs.buf_done
            self.env_step = 0
            filled = np.zeros([self.n_envs, self.episode_length, 1], np.int32)
            rnn_hidden, rnn_hidden_critic = self.init_rnn_hidden()

            if test_mode and self.render:
                images = self.envs.render(self.args.render_mode)
                if self.args.render_mode == "rgb_array":
                    for idx, img in enumerate(images):
                        videos[idx].append(img)

            while not envs_done.all():  # start episodes
                available_actions = self.envs.get_avail_actions()
                actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                                state=state, test_mode=test_mode)
                next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
                envs_done = self.envs.buf_done
                rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']

                if test_mode:
                    if self.render:
                        images = self.envs.render(self.args.render_mode)
                        if self.args.render_mode == "rgb_array":
                            for idx, img in enumerate(images):
                                videos[idx].append(img)
                    for i_env in range(self.n_envs):
                        if terminated[i_env] or truncated[i_env]:  # one env is terminal
                            episode_score.append(info[i_env]["episode_score"])
                            if best_score < episode_score[-1]:
                                best_score = episode_score[-1]
                                episode_videos = videos[i_env].copy()
                            videos[i_env] = []
                else:
                    filled[:, self.env_step] = np.ones([self.n_envs, 1])
                    # store transition data
                    transition = (obs_n, actions_dict, state, rewards, terminated, available_actions)
                    self.agents.memory.store_transitions(self.env_step, *transition)
                    for i_env in range(self.n_envs):
                        if envs_done[i_env]:
                            filled[i_env, self.env_step, 0] = 0
                        else:
                            self.current_step += 1
                        if terminated[i_env] or truncated[i_env]:  # one env is terminal
                            available_actions = self.envs.get_avail_actions()
                            # log
                            if self.use_wandb:
                                step_info["Episode-Steps/env-%d" % i_env] = info[i_env]["episode_step"]
                                step_info["Train-Episode-Rewards/env-%d" % i_env] = info[i_env]["episode_score"]
                            else:
                                step_info["Train-Results/Episode-Steps"] = {"env-%d" % i_env: info[i_env]["episode_step"]}
                                step_info["Train-Results/Episode-Rewards"] = {"env-%d" % i_env: info[i_env]["episode_score"]}
                            self.log_infos(step_info, self.current_step)

                            terminal_data = (next_obs_n, next_state, available_actions, filled)
                            if self.on_policy:
                                if terminated[i_env]:
                                    values_next = np.array([0.0 for _ in range(self.num_agents)])
                                else:
                                    batch_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                                    rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(batch_select,
                                                                                                              *rnn_hidden_critic)
                                    kwargs = {"state": [next_state[i_env]]}
                                    _, values_next = self.agents.values(next_obs_n[i_env:i_env + 1], *rnn_h_critic_i, **kwargs)
                                self.agents.memory.finish_path(i_env, self.env_step+1, *terminal_data,
                                                               value_next=values_next,
                                                               value_normalizer=self.agents.learner.value_normalizer)
                            else:
                                self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data)
                            self.current_step += 1
                    self.env_step += 1
                obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)

            # train the model
            if not test_mode:
                self.agents.memory.store_episodes()  # store episode data
                n_epoch = self.agents.n_epoch if self.on_policy else self.n_envs
                train_info = self.agents.train(self.current_step, n_epoch=n_epoch)  # train
                self.log_infos(train_info, self.current_step)

        # After running n_episodes
        episode_score = np.array(episode_score)
        scores_mean = np.mean(episode_score)
        incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
        incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
        win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
        allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
        incre_allies = float(self.envs.dead_allies_count.sum() - dead_allies)
        incre_enemies = float(self.envs.dead_enemies_count.sum() - dead_enemies)
        dead_ratio = incre_allies / allies_count if allies_count > 0 else 0.0
        enemy_dead_ratio = incre_enemies / enemies_count if enemies_count > 0 else 0.0

        if test_mode:
            if self.args.test_mode:
                print("Mean score: %.4f, Test Win Rate: %.4f." % (scores_mean, win_rate))
            if self.args.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)
            results_info = {"Test-Results/Mean-Episode-Rewards": scores_mean,
                            "Test-Results/Win-Rate": win_rate,
                            "Test-Results/Dead-Ratio": dead_ratio,
                            "Test-Results/Enemy-Dead-Ratio": enemy_dead_ratio}
        else:
            results_info = {"Train-Results/Win-Rate": win_rate,
                            "Train-Results/Dead-Ratio": dead_ratio,
                            "Train-Results/Enemy-Dead-Ratio": enemy_dead_ratio}
        self.log_infos(results_info, self.current_step)

        return episode_score, win_rate

    def run(self):
        if self.args.test_mode:
            self.render = True
            n_test_episodes = self.args.test_episode
            self.agents.load_model(self.args.model_dir_load)
            self.run_episodes(n_test_episodes, test_mode=True)
            print("Finish testing.")
        else:
            n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
            agent_info = f"({self.args.agent}, seed={self.args.seed})"
            time_start = time.time()
            while self.current_step <= self.running_steps:
                print(agent_info, f"Steps: {self.current_step} / {self.running_steps}: ")
                episode_score, win_rate = self.run_episodes(n_train_episodes, test_mode=False)

                # Estimate the physic running time
                mean_scores = episode_score.mean()
                time_pass, time_left = self.time_estimate(time_start)
                print("Win rate: %.3f, Mean score: %.2f. " % (win_rate, mean_scores), time_pass, time_left)

            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def benchmark(self):
        n_eval_interval = self.args.eval_interval // self.n_envs // self.episode_length
        n_test_episodes = self.args.test_episode // self.n_envs

        test_episode_score, best_win_rate = self.run_episodes(n_episodes=n_test_episodes, test_mode=True)
        best_score = {
            "mean": test_episode_score.mean(),
            "std": test_episode_score.std(),
            "step": self.current_step
        }

        agent_info = f"({self.args.agent}, seed={self.args.seed})"
        time_start = time.time()
        while self.current_step <= self.running_steps:
            print(agent_info, f"Steps: {self.current_step} / {self.running_steps}: ")
            # train
            self.run_episodes(n_episodes=n_eval_interval, test_mode=False)
            # test
            test_episode_score, test_win_rate = self.run_episodes(n_episodes=n_test_episodes, test_mode=True)

            mean_test_scores = test_episode_score.mean()
            if best_score["mean"] < mean_test_scores:
                best_score = {
                    "mean": mean_test_scores,
                    "std": test_episode_score.std(),
                    "step": self.current_step
                }
            if best_win_rate < test_win_rate:
                best_win_rate = test_win_rate
                self.agents.save_model("best_model.pth")  # save best model

            # Estimate the physic running time
            time_pass, time_left = self.time_estimate(time_start)
            print("Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, best_score["mean"]), time_pass, time_left)

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: ", best_score["mean"], "Std: ", best_score["std"])
        print("Best Win Rate: {}%".format(best_win_rate * 100))

        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def time_estimate(self, start):
        time_pass = int(time.time() - start)
        time_left = int((self.running_steps - self.current_step) / self.current_step * time_pass)
        if time_left < 0:
            time_left = 0
        hours_pass, hours_left = time_pass // 3600, time_left // 3600
        min_pass, min_left = np.mod(time_pass, 3600) // 60, np.mod(time_left, 3600) // 60
        sec_pass, sec_left = np.mod(np.mod(time_pass, 3600), 60), np.mod(np.mod(time_left, 3600), 60)
        INFO_time_pass = f"Time pass: {hours_pass}h{min_pass}m{sec_pass}s,"
        INFO_time_left = f"Time left: {hours_left}h{min_left}m{sec_left}s"
        return INFO_time_pass, INFO_time_left
