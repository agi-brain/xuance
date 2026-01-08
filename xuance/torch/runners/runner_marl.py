import os
import json
import time
import csv
import copy
import numpy as np
from copy import deepcopy
from argparse import Namespace
from datetime import datetime
from xuance.common import Optional, create_directory
from xuance.torch.runners import RunnerBase
from xuance.torch.agents import REGISTRY_Agents, Agent
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, make_envs


class RunnerMARL(RunnerBase):
    def __init__(self,
                 config: Namespace,
                 envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                 agent: Agent = None,
                 manage_resources: bool = None):
        # Store configuration
        self.config = config
        self.env_id = self.config.env_id
        
        super(RunnerMARL, self).__init__(self.config, envs, agent, manage_resources)
        # Build agent if not injected externally
        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs) if agent is None else agent

        # Distributed training setup (rank-aware behavior)
        if self.agent.distributed_training:
            self.rank = int(os.environ['RANK'])

    def _run_train(self, **kwargs):
        n_train_steps = max(1, self.config.running_steps // self.n_envs)
        self.agent.train(n_train_steps)
        self.rprint("Finish training.")
        self.agent.save_model(model_name="final_train_model.pth")

    def _run_test(self, **kwargs):
        def env_fn():
            config_test = deepcopy(self.config)
            config_test.parallels = 1
            config_test.render = True
            return make_envs(config_test)

        if self.rank == 0:
            self.agent.render = True
            self.agent.load_model(self.agent.model_dir_load)
            scores = self.agent.test(env_fn, self.config.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")

    def _run_benchmark(self, **kwargs):
        running_steps = kwargs.get('running_steps', self.config.running_steps)
        eval_interval = kwargs.get('eval_interval', self.config.eval_interval)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)
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
                header = ["step"] + [f"score_episode_{i}" for i in range(test_episodes)]
                writer.writerow(header)
            with open(learning_curve_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "mean_score"])

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

        # Prepare testing environments.
        config_test = copy.deepcopy(self.config)
        config_test.parallels = 1  # config_test.test_episode
        test_envs = make_envs(config_test)

        train_steps = max(1, running_steps // self.n_envs)
        eval_interval = max(1, eval_interval // self.n_envs)
        num_epoch = train_steps // eval_interval

        # Start benchmarking...
        start_time = time.time()
        start_time_iso = datetime.now().astimezone().isoformat()
        best_model_time_iso = start_time_iso
        if self.rank == 0:
            test_scores = self.agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=False)
            with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step)] + test_scores)
            with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.agent.current_step), np.mean(test_scores)])
        else:
            test_scores = 0.0
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agent.current_step}
        for i_epoch in range(num_epoch):
            self.rprint("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agent.train(train_steps=eval_interval)
            if self.rank == 0:
                test_scores = self.agent.test(test_episodes=test_episodes,
                                              test_envs=test_envs,
                                              close_envs=False)
                with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(self.agent.current_step)] + test_scores)
                with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([int(self.agent.current_step), np.mean(test_scores)])

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agent.current_step}
                    # save best model
                    self.agent.save_model(model_name="best_model.pth", model_path=best_model_path)
                    best_model_time_iso = datetime.now().astimezone().isoformat()

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

        self.agent.save_model(model_name="final_train_model.pth")
        test_envs.close()
        self.rprint("Best Model Score: %.2f, std=%.2f. "
                    "Best Step: %d" % (best_scores_info["mean"], best_scores_info["std"],
                                       best_scores_info["step"]))
