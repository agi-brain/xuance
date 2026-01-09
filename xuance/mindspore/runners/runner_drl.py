import os
import json
import time
import csv
import gymnasium as gym
import numpy as np
from copy import deepcopy
from argparse import Namespace
from datetime import datetime
from xuance.common import Optional, create_directory
from xuance.environment import DummyVecEnv, SubprocVecEnv, make_envs
from xuance.mindspore.runners import RunnerBase
from xuance.mindspore.agents import REGISTRY_Agents, Agent


class RunnerDRL(RunnerBase):
    """Runner for single-agent Deep Reinforcement Learning (DRL).

    RunnerDRL orchestrates the full lifecycle of an experiment, including
    environment creation, agent initialization, training, testing, and
    benchmarking. It is responsible for experiment-level logic rather than
    algorithmic details.

    Responsibilities:
        - Create and manage environments and agent (unless injected externally).
        - Control training, testing, and benchmarking workflows.
        - Handle experiment-level logic (run mode, evaluation loop, model saving).
        - Manage resource lifecycle (envs.close(), agent.finish()) based on
          ownership semantics.

    Notes:
        - Algorithm-specific logic should remain inside the Agent.
        - Runner focuses on experiment orchestration and reproducibility.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
                 agent: Agent = None,
                 manage_resources: bool = None):
        """Initialize the DRL runner.

        This constructor sets up the experiment context. By default, the runner
        creates environments and agent internally using the provided configuration.
        Advanced users may inject pre-created environments or agents.

        Resource ownership is determined automatically unless explicitly specified
        via `manage_resources`.

        Args:
           config: Experiment configuration object. It contains environment,
               agent, and runtime settings.
           envs (optional): Pre-created environments. If None, environments will
               be created internally by the runner.
           agent (optional): Pre-created agent. If None, the runner will build
               the agent using the registry.
           manage_resources (optional): Whether the runner is responsible for
               closing environments and finalizing the agent.
               - If None, ownership is inferred automatically.
               - If True, runner will call envs.close() and agent.finish().
               - If False, resource lifecycle is managed externally.
        """
        # Store configuration
        self.config = config
        self.env_id = self.config.env_id

        super(RunnerDRL, self).__init__(self.config, envs, agent, manage_resources)

        if self.env_id in ['Platform-v0']:
            self.config.observation_space = self.envs.observation_space.spaces[0]
            old_as = self.envs.action_space
            num_disact = old_as.spaces[0].n
            self.config.action_space = gym.spaces.Tuple(
                (old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                    old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in
                                     range(0, num_disact))))
        else:
            self.config.observation_space = self.envs.observation_space
            self.config.action_space = self.envs.action_space

        # Build agent if not injected externally
        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs) if agent is None else agent

        # Distributed training setup (rank-aware behavior)
        if self.agent.distributed_training:
            self.rank = int(os.environ['RANK'])

    def _run_train(self, **kwargs):
        running_steps = kwargs.get('running_steps', self.config.running_steps)
        n_train_steps = max(1, running_steps // self.n_envs)
        self.agent.train(n_train_steps)
        print("Finish training.")
        self.agent.save_model(model_name="final_train_model")

    def _run_test(self, **kwargs):
        config_test = deepcopy(self.config)
        config_test.parallels = kwargs.get("n_envs", 1)
        config_test.render = self.agent.render = kwargs.get('render', True)
        model_path = kwargs.get('model_path', self.agent.model_dir_load)
        test_episodes = kwargs.get('test_episodes', self.config.test_episode)
        test_envs = make_envs(config_test)

        if self.rank == 0:
            self.agent.load_model(model_path)
            scores = self.agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=True)
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
        with open(test_scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["step"] + [f"return_episode_{i}" for i in range(test_episodes)]
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

        # Prepare testing environments.
        config_test = deepcopy(self.config)
        config_test.parallels = test_episodes
        test_envs = make_envs(config_test)

        train_steps = max(1, running_steps // self.n_envs)
        eval_interval = max(1, eval_interval // self.n_envs)
        num_epoch = train_steps // eval_interval

        # Start benchmarking...
        start_time = time.time()
        start_time_iso = datetime.now().astimezone().isoformat()
        best_model_time_iso = start_time_iso
        test_scores = self.agent.test(test_episodes=test_episodes, test_envs=test_envs, close_envs=False)

        with open(test_scores_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([int(self.agent.current_step)] + test_scores)
        with open(learning_curve_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([int(self.agent.current_step), np.mean(test_scores)])

        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agent.train(train_steps=eval_interval)
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
                self.agent.save_model(model_name="best_model", model_path=best_model_path)
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
        with open(os.path.join(benchmark_result_path, "meta_data.json"), "w", encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(benchmark_result_path, "config.json"), "w", encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        with open(os.path.join(benchmark_result_path, "best_model_info.json"), "w", encoding='utf-8') as f:
            json.dump(best_scores_info, f, indent=2, ensure_ascii=False)

        self.agent.save_model(model_name="final_train_model")
        test_envs.close()
        print("Best Model Score: %.2f, std=%.2f. "
              "Best Step: %d" % (best_scores_info["mean"], best_scores_info["std"],
                                 best_scores_info["step"]))
