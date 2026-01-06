import os
import copy
import numpy as np
from copy import deepcopy
from argparse import Namespace
from xuance.common import Optional
from xuance.torch.runners import RunnerBase
from xuance.torch.agents import REGISTRY_Agents, Agent
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, make_envs


class RunnerMARL(RunnerBase):
    def __init__(self, config: Namespace,
                 envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                 agent: Agent = None,
                 manage_resources: bool = None):
        # Store configuration
        self.config = config
        self.env_id = self.config.env_id
        
        super(RunnerMARL, self).__init__(self.config, envs, agent, manage_resources)
        # Build agent if not injected externally
        self.agents = REGISTRY_Agents[self.config.agent](self.config, self.envs) if agent is None else agent

        # Distributed training setup (rank-aware behavior)
        if self.agents.distributed_training:
            self.rank = int(os.environ['RANK'])

    def _run_train(self, **kwargs):
        n_train_steps = max(1, self.config.running_steps // self.n_envs)
        self.agents.train(n_train_steps)
        self.rprint("Finish training.")
        self.agents.save_model(model_name="final_train_model.pth")

    def _run_test(self, **kwargs):
        def env_fn():
            config_test = deepcopy(self.config)
            config_test.parallels = 1
            config_test.render = True
            return make_envs(config_test)

        if self.rank == 0:
            self.agents.render = True
            self.agents.load_model(self.agents.model_dir_load)
            scores = self.agents.test(env_fn, self.config.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")

    def _run_benchmark(self):
        def env_fn():
            config_test = copy.deepcopy(self.config)
            config_test.parallels = 1  # config_test.test_episode
            return make_envs(config_test)

        train_steps = self.config.running_steps // self.n_envs
        eval_interval = self.config.eval_interval // self.n_envs
        test_episode = self.config.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.agents.test(env_fn, test_episode) if self.rank == 0 else 0.0
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agents.train(eval_interval)
            if self.rank == 0:
                test_scores = self.agents.test(env_fn, test_episode)

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agents.current_step}
                    # save best model
                    self.agents.save_model(model_name="best_model.pth")

        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        self.agents.finish()
        self.envs.close()
