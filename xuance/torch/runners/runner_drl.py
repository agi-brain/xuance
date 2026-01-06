import os
import gymnasium as gym
import numpy as np
from copy import deepcopy
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv, make_envs
from xuance.torch.runners import RunnerBase
from xuance.torch.agents import REGISTRY_Agents, Agent


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

    def _run_benchmark(self):
        def env_fn():
            config_test = deepcopy(self.config)
            config_test.parallels = config_test.test_episode
            return make_envs(config_test)

        train_steps = max(1, self.config.running_steps // self.n_envs)
        eval_interval = max(1, self.config.eval_interval // self.n_envs)
        test_episode = self.config.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.agent.test(env_fn, test_episode) if self.rank == 0 else 0.0
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agent.current_step}
        for i_epoch in range(num_epoch):
            self.rprint("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agent.train(eval_interval)
            if self.rank == 0:
                test_scores = self.agent.test(env_fn, test_episode)

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agent.current_step}
                    # save best model
                    self.agent.save_model(model_name="best_model.pth")

        # end benchmarking
        self.agent.save_model(model_name="final_train_model.pth")
        self.rprint("Best Model Score: %.2f, std=%.2f. "
                    "Best Step: %d" % (best_scores_info["mean"], best_scores_info["std"],
                                       best_scores_info["step"]))

