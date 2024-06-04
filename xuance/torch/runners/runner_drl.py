import gym.spaces
import numpy as np
from copy import deepcopy
from xuance.torch.runners import Runner_Base
from xuance.environment import make_envs
from xuance.torch.agents import REGISTRY_Agents


class Runner_DRL(Runner_Base):
    def __init__(self, config):
        self.config = config
        self.env_id = self.config.env_id
        super(Runner_DRL, self).__init__(self.config)

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

        self.agent = REGISTRY_Agents[self.config.agent](self.config, self.envs)

    def run(self):
        if self.config.test_mode:
            def env_fn():
                config_test = deepcopy(self.config)
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)

            self.agent.render = True
            self.agent.load_model(self.agent.model_dir_load)
            scores = self.agent.test(env_fn, self.config.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            n_train_steps = self.config.running_steps // self.n_envs
            self.agent.train(n_train_steps)
            print("Finish training.")
            self.agent.save_model("final_train_model.pth")

        self.envs.close()
        self.agent.finish()

    def benchmark(self):
        # test environment
        def env_fn():
            config_test = deepcopy(self.config)
            config_test.parallels = config_test.test_episode
            return make_envs(config_test)

        train_steps = self.config.running_steps // self.n_envs
        eval_interval = self.config.eval_interval // self.n_envs
        test_episode = self.config.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = self.agent.test(env_fn, test_episode)
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": self.agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agent.train(eval_interval)
            test_scores = self.agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": self.agent.current_step}
                # save best model
                self.agent.save_model(model_name="best_model.pth")

        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))

        self.envs.close()
        self.agent.finish()
