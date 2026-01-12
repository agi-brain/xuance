import os
import numpy as np
from copy import deepcopy

from xuance.environment import make_envs
from xuance.torch.runners import RunnerBase
from xuance.torch.agents import REGISTRY_Agents
try:
    from xuance.common import load_d4rl_dataset
except:
    pass


class RunnerOfflineRL(RunnerBase):
    def __init__(self, config):
        self.config = config
        self.env_id = self.config.env_id
        super(RunnerOfflineRL, self).__init__(self.config)

        self.config.observation_space = self.envs.observation_space
        self.config.action_space = self.envs.action_space

        dataset, state_mean, state_std = load_d4rl_dataset(
            dataset_name=config.dataset,
            max_episode_steps=config.test_steps // config.test_episode,
            obsnorm=config.normalize_obs_offline,
            rewnorm=config.normalize_reward_offline
        )
        self.config.state_mean = state_mean
        self.config.state_std = state_std

        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs)
        self.agent.load_dataset(dataset=dataset)
        if self.agent.distributed_training:
            self.rank = int(os.environ['RANK'])

    def run(self):
        if self.config.test_mode:
            def env_fn():
                self.config.parallels = self.config.test_episode
                return make_envs(self.config)

            self.agent.load_model(self.agent.model_dir_load)
            scores = self.agent.test(env_fn, self.config.test_episode)
            self.rprint(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            self.rprint("Finish testing.")
        else:
            self.agent.train(self.config.running_steps)
            self.agent.save_model("final_train_model.pth")
            self.rprint("Finish training.")
        self.agent.finish()
        self.envs.close()

    def benchmark(self):
        # test environment
        def env_fn():
            config_test = deepcopy(self.config)
            config_test.parallels = config_test.test_episode
            return make_envs(config_test)

        eval_interval = self.config.eval_interval
        test_episode = self.config.test_episode
        num_epoch = int(self.config.running_steps / eval_interval)

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
                print(f"Normalized-Test-Episode-Rewards: {test_scores}")
                print(f"D4RL-Score: %.3f" % np.mean(test_scores))

        # end benchmarking
        self.rprint("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        self.agent.finish()
        self.envs.close()
