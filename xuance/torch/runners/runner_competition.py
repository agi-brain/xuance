import os
import argparse
from copy import deepcopy
import numpy as np
from xuance.torch.agents import REGISTRY_Agents
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed


class RunnerCompetition(object):
    def __init__(self, configs):
        self.configs = configs
        # set random seeds
        set_seed(configs.seed)

        # build environments
        self.envs = make_envs(self.configs[0])
        self.envs.reset()
        self.group_info = self.envs.groups_infos
        self.groups = self.group_info['agent_groups']
        self.num_groups = self.group_info['num_groups']
        self.obs_space_groups = self.group_info['observation_space_groups']
        self.act_space_groups = self.group_info['action_space_groups']
        assert len(configs) == self.num_groups, "Number of groups must be equal to the number of methods."
        self.agents = []
        for group in range(self.num_groups):
            _env_info = dict(num_agents=len(self.groups[group]),
                             num_envs=self.envs.num_envs,
                             agents=self.groups[group],
                             state_space=self.envs.state_space,
                             observation_space=self.obs_space_groups[group],
                             action_space=self.act_space_groups[group],
                             max_episode_steps=self.envs.max_episode_steps)
            _env = argparse.Namespace(**_env_info)
            self.agents.append(REGISTRY_Agents[self.configs[group].agent](self.configs[group], _env))

        self.observation_space = self.envs.observation_space
        self.n_envs = self.envs.num_envs
        self.rank = 0
        if self.agents[0].distributed_training:
            self.rank = int(os.environ['RANK'])

    def rprint(self, info: str):
        if self.rank == 0:
            print(info)

    def run(self):
        if self.configs[0].test_mode:
            def env_fn():
                config_test = deepcopy(self.configs[0])
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)

            for agent in self.agents:
                agent.render = True
                agent.load_model(agent.model_dir_load)

            # ... Here is test ...
            scores = self.agents.test(env_fn, self.config.test_episode)

            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            n_train_steps = self.configs[0].running_steps // self.n_envs

            # ... Here is train ...
            self.agents.train(n_train_steps)

            print("Finish training.")
            for agent in self.agents:
                agent.save_model("final_train_model.pth")

        for agent in self.agents:
            agent.finish()

    def benchmark(self):
        def env_fn():
            config_test = deepcopy(self.configs[0])
            config_test.parallels = 1  # config_test.test_episode
            return make_envs(config_test)

        train_steps = self.configs[0].running_steps // self.n_envs
        eval_interval = self.configs[0].eval_interval // self.n_envs
        test_episode = self.configs[0].test_episode
        num_epoch = int(train_steps / eval_interval)

        # ... Here is test ...
        test_scores = self.agents.test(env_fn, test_episode) if self.rank == 0 else 0.0

        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agents[0].current_step}

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))

            # ... Here is train ...
            self.agents.train(eval_interval)

            if self.rank == 0:

                # ... Here is test ...
                test_scores = self.agents.test(env_fn, test_episode)

                if np.mean(test_scores) > best_scores_info["mean"]:
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": self.agents.current_step}
                    # save best model
                    for agent in self.agents:
                        agent.save_model(model_name="best_model.pth")

        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        for agent in self.agents:
            agent.finish()

    def train(self, eval_interval):
        return

    def test(self, env_fn, test_episode):
        scores = [None for handel in self.handles]
        return scores
