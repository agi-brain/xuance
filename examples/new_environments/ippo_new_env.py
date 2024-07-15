import argparse
import numpy as np
from copy import deepcopy
from gym.spaces import Box
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawMultiAgentEnv, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import IPPO_Agents


class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        self.env_id = env_config.env_id
        self.num_agents = 3
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(-np.inf, np.inf, shape=[8, ])
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[8, ]) for agent in self.agents}
        self.action_space = {agent: Box(-np.inf, np.inf, shape=[2, ]) for agent in self.agents}
        self.max_episode_steps = 25
        self._current_step = 0

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def reset(self):
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        info = {}
        self._current_step = 0
        return observation, info

    def step(self, action_dict):
        self._current_step += 1
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        rewards = {agent: np.random.random() for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: IPPO for MPE.")
    parser.add_argument("--env-id", type=str, default="new_env_id")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="new_configs/ippo_new_configs.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv
    set_seed(configs.seed)  # Set the random seed.
    envs = make_envs(configs)  # Make the environment.
    Agents = IPPO_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():  # Print the training information.
        print(f"{k}: {v}")

    if configs.benchmark:
        def env_fn():  # Define an environment function for test method.
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)


        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agents.test(env_fn, test_episode)
        Agents.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agents.train(eval_interval)
            test_scores = Agents.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agents.current_step}
                # save best model
                Agents.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)


            Agents.load_model(path=Agents.model_dir_load)
            scores = Agents.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agents.train(configs.running_steps // configs.parallels)
            Agents.save_model("final_train_model.pth")
            print("Finish training!")

    Agents.finish()
