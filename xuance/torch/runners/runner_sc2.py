import os
import json
import time
import csv
import numpy as np
from argparse import Namespace
from datetime import datetime
from xuance.common import Optional, create_directory
from xuance.torch.runners import RunnerBase
from xuance.torch.agents import REGISTRY_Agents, Agent
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, make_envs


class RunnerSC2(RunnerBase):
    def __init__(self,
                 config: Namespace,
                 envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                 agent: Agent = None,
                 manage_resources: bool = None):
        # Store configuration
        self.config = config
        self.env_id = self.config.env_id
        self.running_steps = config.running_steps

        super(RunnerSC2, self).__init__(self.config, envs, agent, manage_resources)
        self.config.n_agents = self.envs.num_agents
        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs) if agent is None else agent
        self.num_agents, self.num_enemies = self.get_agent_num()

        # Distributed training setup (rank-aware behavior)
        if self.agent.distributed_training:
            self.rank = int(os.environ['RANK'])

    def get_agent_num(self):
        return self.envs.num_agents, self.envs.num_enemies

    def get_battles_info(self):
        battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
        dead_allies, dead_enemies = self.envs.dead_allies_count.sum(), self.envs.dead_enemies_count.sum()
        return battles_game, battles_won, dead_allies, dead_enemies

    def get_battles_result(self, last_battles_info):
        battles_game, battles_won, dead_allies, dead_enemies = list(last_battles_info)
        incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
        incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
        win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
        allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
        incre_allies = float(self.envs.dead_allies_count.sum() - dead_allies)
        incre_enemies = float(self.envs.dead_enemies_count.sum() - dead_enemies)
        allies_dead_ratio = incre_allies / allies_count if allies_count > 0 else 0.0
        enemies_dead_ratio = incre_enemies / enemies_count if enemies_count > 0 else 0.0
        return win_rate, allies_dead_ratio, enemies_dead_ratio

    def test_episodes(self, test_T, n_test_runs):
        test_scores = np.zeros(n_test_runs, np.float32)
        last_battles_info = self.get_battles_info()
        for i_test in range(n_test_runs):
            running_scores = self.agent.run_episodes(
                n_episodes=self.n_envs,  # Number of testing episodes 
                run_envs=None,  # The running envs. If None, `self.agent.train_envs` is used. 
                test_mode=True,  # Test mode is on.
                close_envs=False  # Don't close the testing envs (self.agent.train_envs).
            )
            test_scores[i_test] = np.mean(running_scores)
        win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
        mean_test_score = test_scores.mean()
        results_info = {"Test-Results/Win-Rate": win_rate,
                        "Test-Results/Allies-Dead-Ratio": allies_dead_ratio,
                        "Test-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
        self.agent.log_infos(results_info, test_T)
        return test_scores, mean_test_score, test_scores.std(), win_rate

    def _run_train(self, **kwargs):
        eval_interval = self.config.eval_interval
        last_test_T = 0
        episode_scores = []
        agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
        self.rprint(f"Steps: {self.agent.current_step} / {self.running_steps}: ")
        self.rprint(agent_info, "Win rate: %-, Mean score: -.")
        last_battles_info = self.get_battles_info()
        time_start = time.time()
        while self.agent.current_step <= self.running_steps:
            score = self.agent.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
            if self.agent.current_step >= self.agent.start_training:
                train_info = self.agent.train_epochs(n_epochs=1)
                self.agent.log_infos(train_info, self.agent.current_step)
            episode_scores.append(np.mean(score))
            if self.rank == 0:
                if (self.agent.current_step - last_test_T) / eval_interval >= 1.0:
                    last_test_T += eval_interval
                    # log train results before testing.
                    train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                    results_info = {"Train-Results/Win-Rate": train_win_rate,
                                    "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                    "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                    self.agent.log_infos(results_info, last_test_T)
                    last_battles_info = self.get_battles_info()
                    time_pass, time_left = self.time_estimate(time_start)
                    print(f"Steps: {self.agent.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (train_win_rate, np.mean(episode_scores)),
                          time_pass, time_left)
                    episode_scores = []

        self.rprint("Finish training.")
        self.agent.save_model("final_train_model.pth")

    def _run_test(self, **kwargs):
        model_path = kwargs.get('model_path', self.agent.model_dir_load)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)

        if self.rank == 0:
            self.agent.load_model(model_path)
            _, test_score_mean, test_score_std, test_win_rate = self.test_episodes(0, test_episodes)
            agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
            print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
            print("Finish testing.")

    def _run_benchmark(self, **kwargs):
        running_steps = kwargs.get('running_steps', self.config.running_steps)
        eval_interval = kwargs.get('eval_interval', self.config.eval_interval)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)
        n_test_runs = test_episodes // self.n_envs

        benchmark_result_path = kwargs.get('benchmark_result_path', self.config.result_dir)
        best_model_path = os.path.join(os.getcwd(), benchmark_result_path, "best_model")
        # Prepare directory for storing benchmark results.
        benchmark_result_path = os.path.join(os.getcwd(), benchmark_result_path)
        create_directory(benchmark_result_path)
        # Create test_scores.csv file to store testing scores.
        test_scores_csv = os.path.join(benchmark_result_path, "test_scores.csv")
        learning_curve_csv = os.path.join(benchmark_result_path, "learning_curve.csv")
        if self.rank == 0:
            with open(test_scores_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["step"] + [f"return_episode_{i}" for i in range(self.config.test_episode)]
                writer.writerow(header)
            with open(learning_curve_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "avg_return"])

        meta_data = self.agent.meta_data
        meta_data['benchmark'] = {
            "running_steps": running_steps,
            "eval_interval": eval_interval,
            "test_episodes": test_episodes,
        }
        meta_data["system_info"] = self.collect_device_info()
        config_dict = vars(self.agent.config).copy()
        config_dict.pop("observation_space", None)
        config_dict.pop("action_space", None)

        # Start benchmarking...
        start_time = time.time()
        start_time_iso = datetime.now().astimezone().isoformat()
        best_model_time_iso = start_time_iso
        if self.rank == 0:
            # test the model at step 0
            test_scores, test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)
            with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step)] + test_scores)
            with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step), np.mean(test_scores)])
            best_win_rate = test_win_rate
        else:
            best_win_rate = 0
            test_score_std = 0
        last_test_T = 0
        agent_info = f"Algo: {self.config.agent}, Map: {self.config.env_id}, seed: {self.config.seed}, "
        self.rprint(f"Steps: {self.agent.current_step} / {self.running_steps}: ")
        self.rprint(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
        last_battles_info = self.get_battles_info()
        best_scores_info = {"mean": test_score_mean,
                            "std": test_score_std,
                            "step": self.agent.current_step}
        while self.agent.current_step <= self.running_steps:
            # train
            self.agent.run_episodes(
                n_episodes=self.n_envs,  # Number of testing episodes 
                run_envs=None,  # The running envs. If None, `self.agent.train_envs` is used. 
                test_mode=False,  # Test mode is on.
                close_envs=False  # Don't close the testing envs (self.agent.train_envs).
            )
            if self.agent.current_step >= self.agent.start_training:
                train_info = self.agent.train_epochs(n_epochs=self.n_envs)
                self.agent.log_infos(train_info, self.agent.current_step)
            # test
            if self.rank == 0:
                if (self.agent.current_step - last_test_T) / eval_interval >= 1.0:
                    last_test_T += eval_interval
                    # log train results before testing.
                    train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                    results_info = {"Train-Results/Win-Rate": train_win_rate,
                                    "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                    "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                    self.agent.log_infos(results_info, last_test_T)

                    # test the model
                    test_scores, test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T,
                                                                                                     n_test_runs)

                    if test_score_mean > best_scores_info["mean"]:
                        best_scores_info = {"mean": test_score_mean,
                                            "std": test_score_std,
                                            "step": self.agent.current_step}
                    if test_win_rate > best_win_rate:
                        best_win_rate = test_win_rate
                        self.agent.save_model("best_model.pth", model_path=best_model_path)  # save best model
                        best_model_time_iso = datetime.now().astimezone().isoformat()

                    last_battles_info = self.get_battles_info()

                    # Estimate the physic running time
                    time_pass, time_left = self.time_estimate(start_time)
                    print(f"Steps: {self.agent.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean),
                          time_pass, time_left)

        # End benchmarking.
        # Save best model information.
        end_time = time.time()
        end_time_iso = datetime.now().astimezone().isoformat()
        timestamps = {
            "start_time": start_time_iso,
            "best_model_time": best_model_time_iso,
            "end_time": end_time_iso,
            "elapsed_seconds": round(end_time - start_time, 3)
        }
        meta_data["timestamps"] = timestamps
        if self.rank == 0:
            with open(os.path.join(benchmark_result_path, "meta_data.json"), "w", encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            with open(os.path.join(benchmark_result_path, "config.json"), "w", encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            with open(os.path.join(benchmark_result_path, "best_model_info.json"), "w", encoding='utf-8') as f:
                json.dump(best_scores_info, f, indent=2, ensure_ascii=False)

        self.rprint("Finish benchmarking.")
        self.rprint("Best Score: %.4f, Std: %.4f. "
                    "Best Step: %d" % (best_scores_info["mean"], best_scores_info["std"],
                                       best_scores_info['step']))
        self.rprint("Best Win Rate: {}%".format(best_win_rate * 100))

    def time_estimate(self, start):
        current_step = self.agent.current_step
        time_pass = int(time.time() - start)
        time_left = int((self.running_steps - current_step) / current_step * time_pass)
        if time_left < 0:
            time_left = 0
        hours_pass, hours_left = time_pass // 3600, time_left // 3600
        min_pass, min_left = np.mod(time_pass, 3600) // 60, np.mod(time_left, 3600) // 60
        sec_pass, sec_left = np.mod(np.mod(time_pass, 3600), 60), np.mod(np.mod(time_left, 3600), 60)
        INFO_time_pass = f"Time pass: {hours_pass}h{min_pass}m{sec_pass}s,"
        INFO_time_left = f"Time left: {hours_left}h{min_left}m{sec_left}s"
        return INFO_time_pass, INFO_time_left
