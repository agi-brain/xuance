import os
import socket
import time
from pathlib import Path
import wandb
import argparse
from copy import deepcopy
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.common import get_time_string


def parse_args():
    parser = argparse.ArgumentParser("Example: MAPPO of XuanCe for Google Football Research environments.")
    parser.add_argument("--method", type=str, default="mappo")
    parser.add_argument("--env", type=str, default="football")
    parser.add_argument("--env-id", type=str, default="3v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default=f"./mappo_football_configs/{parser.parse_args().env_id}.yaml")

    return parser.parse_args()


class Runner():
    def __init__(self, args):
        # Set random seeds
        set_seed(args.seed)

        # Prepare directories
        self.args = args
        self.args.agent_name = args.agent
        time_string = get_time_string()
        folder_name = f"seed_{args.seed}_" + time_string
        self.args.model_dir_load = self.args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), self.args.model_dir, folder_name)

        if args.test:
            args.parallels = 1
            self.render = args.render = True
        else:
            self.render = args.render = False

        # Logger
        if self.args.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), self.args.log_dir, folder_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        else:
            config_dict = vars(self.args)
            wandb_dir = Path(os.path.join(os.getcwd(), self.args.log_dir))
            if not wandb_dir.exists():
                os.makedirs(str(wandb_dir))
            wandb.init(config=config_dict,
                       project=self.args.project_name,
                       entity=self.args.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=self.args.env_id,
                       job_type=self.args.agent,
                       name=time_string,
                       reinit=True)
            self.use_wandb = True

        # Build environments
        self.envs = make_envs(args)
        self.envs.reset()

        # Running details
        self.running_steps = args.running_steps
        self.training_frequency = args.training_frequency
        self.current_step = 0
        self.env_step = 0
        self.current_episode = np.zeros((self.envs.num_envs,), np.int32)

        # Environment details.
        self.n_envs = self.envs.num_envs
        self.fps = 20
        self.episode_length = self.envs.max_episode_length
        args.n_agents = self.num_agents = self.envs.num_agents
        self.num_adversaries = self.envs.num_adversaries
        self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
        args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
        args.obs_shape, args.act_shape = (self.dim_obs,), ()
        args.rew_shape = args.done_shape = (1,)
        args.action_space = self.envs.action_space
        args.state_space = self.envs.state_space

        # Create MAPPO agents
        from xuance.torch.agents import MAPPO_Agents
        self.agents = MAPPO_Agents(args, self.envs, args.device)

    def log_infos(self, info: dict, x_index: int):
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
        rnn_hidden_policy, rnn_hidden_critic = rnn_hidden[0], rnn_hidden[1]
        rnn_hidden_next, actions_n, log_pi_n = self.agents.act(obs_n, *rnn_hidden_policy, avail_actions=avail_actions,
                                                               test_mode=test_mode)
        if test_mode:
            rnn_hidden_critic_next, values_n = None, 0
        else:
            rnn_hidden_critic_next, values_n = self.agents.values(obs_n, *rnn_hidden_critic, state=state)
        return {'actions_n': actions_n, 'log_pi': log_pi_n, 'values': values_n,
                'rnn_hidden': rnn_hidden_next, 'rnn_hidden_critic': rnn_hidden_critic_next}

    def get_battles_info(self):
        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        return battles_game, battles_won

    def get_battles_result(self, last_battles_info):
        battles_game, battles_won = list(last_battles_info)
        incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
        incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
        win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
        return win_rate

    def run_episodes(self, test_mode=False):
        episode_score, episode_step, best_score = [], [], -np.inf

        # reset the envs and settings
        obs_n, state, info = self.envs.reset()
        envs_done = self.envs.buf_done
        self.env_step = 0
        filled = np.zeros([self.n_envs, self.episode_length, 1], np.int32)
        rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs * self.num_agents)
        rnn_hidden_critic = self.agents.policy.representation_critic.init_hidden(self.n_envs * self.num_agents)

        while not envs_done.all():  # start episodes
            available_actions = self.envs.get_avail_actions()
            actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                            state=state, test_mode=test_mode)
            next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
            envs_done = self.envs.buf_done
            rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']

            if test_mode:
                for i_env in range(self.n_envs):
                    if terminated[i_env] or truncated[i_env]:  # one env is terminal
                        episode_score.append(info[i_env]["episode_score"])
                        if best_score < episode_score[-1]:
                            best_score = episode_score[-1]
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
                        episode_score.append(info[i_env]["episode_score"])
                        episode_step.append(info[i_env]["episode_step"])
                        available_actions = self.envs.get_avail_actions()

                        terminal_data = (next_obs_n, next_state, available_actions, filled)
                        if terminated[i_env]:
                            values_next = np.array([0.0 for _ in range(self.num_agents)])
                        else:
                            batch_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                            kwargs = {"state": [next_state[i_env]]}
                            rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(batch_select,
                                                                                                      *rnn_hidden_critic)
                            _, values_next = self.agents.values(next_obs_n[i_env:i_env + 1],
                                                                *rnn_h_critic_i, **kwargs)
                        self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data,
                                                       value_next=values_next,
                                                       value_normalizer=self.agents.learner.value_normalizer)
                        self.current_step += 1
                self.env_step += 1
            obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)

        if not test_mode:
            self.agents.memory.store_episodes()  # store episode data
            n_epoch = self.agents.n_epoch
            train_info = self.agents.train(self.current_step, n_epoch=n_epoch)  # train
            train_info["Train-Results/Train-Episode-Rewards"] = np.mean(episode_score)
            train_info["Train-Results/Episode-Steps"] = np.mean(episode_step)
            self.log_infos(train_info, self.current_step)

        mean_episode_score = np.mean(episode_score)
        return mean_episode_score

    def test_episodes(self, test_T, n_test_runs):
        test_scores = np.zeros(n_test_runs)
        last_battles_info = self.get_battles_info()
        for i_test in range(n_test_runs):
            test_scores[i_test] = self.run_episodes(test_mode=True)
        win_rate = self.get_battles_result(last_battles_info)
        mean_test_score = test_scores.mean()
        results_info = {"Test-Results/Mean-Episode-Rewards": mean_test_score,
                        "Test-Results/Win-Rate": win_rate}
        self.log_infos(results_info, test_T)
        return mean_test_score, test_scores.std(), win_rate

    def run(self):
        if self.args.test_mode:
            n_test_episodes = self.args.test_episode
            self.agents.load_model(self.args.model_dir_load)
            test_score_mean, test_score_std, test_win_rate = self.test_episodes(0, n_test_episodes)
            agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
            print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
            print("Finish testing.")
        else:
            test_interval = self.args.eval_interval
            last_test_T = 0
            episode_scores = []
            agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
            print(f"Steps: {self.current_step} / {self.running_steps}: ")
            print(agent_info, "Win rate: %-, Mean score: -.")
            last_battles_info = self.get_battles_info()
            time_start = time.time()
            while self.current_step <= self.running_steps:
                score = self.run_episodes(test_mode=False)
                episode_scores.append(score)
                if (self.current_step - last_test_T) / test_interval >= 1.0:
                    last_test_T += test_interval
                    # log train results before testing.
                    train_win_rate = self.get_battles_result(last_battles_info)
                    results_info = {"Train-Results/Win-Rate": train_win_rate}
                    self.log_infos(results_info, last_test_T)
                    last_battles_info = self.get_battles_info()
                    time_pass, time_left = self.time_estimate(time_start)
                    print(f"Steps: {self.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (train_win_rate, np.mean(episode_scores)),
                          time_pass, time_left)
                    episode_scores = []

            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

    def benchmark(self):
        test_interval = self.args.eval_interval
        n_test_runs = self.args.test_episode // self.n_envs
        last_test_T = 0

        # test the mode at step 0
        test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)
        best_score = {"mean": test_score_mean,
                      "std": test_score_std,
                      "step": self.current_step}
        best_win_rate = test_win_rate

        agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
        print(f"Steps: {self.current_step} / {self.running_steps}: ")
        print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
        last_battles_info = self.get_battles_info()
        time_start = time.time()
        while self.current_step <= self.running_steps:
            # train
            self.run_episodes(test_mode=False)
            # test
            if (self.current_step - last_test_T) / test_interval >= 1.0:
                last_test_T += test_interval
                # log train results before testing.
                train_win_rate = self.get_battles_result(last_battles_info)
                results_info = {"Train-Results/Win-Rate": train_win_rate}
                self.log_infos(results_info, last_test_T)

                # test the model
                test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)

                if best_score["mean"] < test_score_mean:
                    best_score = {"mean": test_score_mean,
                                  "std": test_score_std,
                                  "step": self.current_step}
                if best_win_rate < test_win_rate:
                    best_win_rate = test_win_rate
                    self.agents.save_model("best_model.pth")  # save best model

                last_battles_info = self.get_battles_info()

                # Estimate the physic running time
                time_pass, time_left = self.time_estimate(time_start)
                print(f"Steps: {self.current_step} / {self.running_steps}: ")
                print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean), time_pass, time_left)

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score: %.4f, Std: %.4f" % (best_score["mean"], best_score["std"]))
        print("Best Win Rate: {}%".format(best_win_rate * 100))

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

    def finish(self):
        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser,
                         is_test=parser.test)
    runner = Runner(args)
    runner.benchmark()
    runner.finish()
