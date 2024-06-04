import copy
import numpy as np
from xuance.torch.runners import Runner_Base
from xuance.torch.agents import REGISTRY_Agents
from xuance.environment import make_envs


class Runner_MARL(Runner_Base):
    def __init__(self, config):
        super(Runner_MARL, self).__init__(config)
        config.n_agents = self.envs.num_agents
        self.agents = REGISTRY_Agents[config.agent](config, self.envs)
        self.config = config

    def run(self):
        if self.config.test_mode:
            def env_fn():
                config_test = copy.deepcopy(self.config)
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)
            self.agents.render = True
            self.agents.load_model(self.agents.model_dir_load)
            scores = self.agents.test(env_fn, self.config.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            n_train_steps = self.config.running_steps // self.n_envs
            self.agents.train(n_train_steps)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

        self.envs.close()
        self.agents.finish()

    def benchmark(self):
        def env_fn():
            config_test = copy.deepcopy(self.config)
            config_test.parallels = 1  # config_test.test_episode
            return make_envs(config_test)

        train_steps = self.config.running_steps // self.n_envs
        eval_interval = self.config.eval_interval // self.n_envs
        test_episode = self.config.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.agents.test(env_fn, test_episode)
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agents.train(eval_interval)
            test_scores = self.agents.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": self.agents.current_step}
                # save best model
                self.agents.save_model(model_name="best_model.pth")

        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        self.envs.close()
        self.agents.finish()
