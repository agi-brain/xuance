import os
import argparse
import numpy as np
from copy import deepcopy
from gymnasium.spaces import Box, Discrete
from xuance.common import get_configs, recursive_dict_update, create_directory
from xuance.environment import make_envs, RawEnvironment, REGISTRY_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import DQN_Agent, BaseCallback
from torch.utils.tensorboard import SummaryWriter


class MyNewEnv(RawEnvironment):
    "The custom environment."
    def __init__(self, env_config):
        super(MyNewEnv, self).__init__()
        self.env_id = env_config.env_id
        self.observation_space = Box(-np.inf, np.inf, shape=[18, ])
        self.action_space = Discrete(n=5)
        self.max_episode_steps = 32
        self._current_step = 0

    def reset(self, **kwargs):
        self._current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._current_step += 1
        observation = self.observation_space.sample()
        rewards = np.random.random()
        terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {
            "info_1": np.random.rand(),
            "info_2": np.random.rand(),
        }
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return


class MyCallback(BaseCallback):
    """The custom callback.
        Note: Defining a custom callback is not required to use the agent.
        This example is provided purely to illustrate how one might extend the training loop with user-defined logic.
    """
    def __init__(self, config):
        super(MyCallback, self).__init__()
        log_dir = os.path.join(os.getcwd(), config.log_dir, 'callback_info')
        create_directory(log_dir)
        self.writer = SummaryWriter(log_dir)

    def on_train_episode_info(self, *args, **kwargs):
        """Visualize the additional information about the environment on Tensorboard."""
        infos = kwargs['infos']
        env_id = kwargs['env_id']
        step = kwargs['current_step']
        self.writer.add_scalars('environment_information/info_1', {f"env-{env_id}": infos[env_id]["info_1"]}, step)
        self.writer.add_scalars('environment_information/info_2', {f"env-{env_id}": infos[env_id]["info_2"]}, step)


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DQN.")
    parser.add_argument("--env-id", type=str, default="new_env_id")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="new_configs/dqn_new_env.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_ENV[configs.env_name] = MyNewEnv  # Register the new environment.
    set_seed(configs.seed)
    envs = make_envs(configs)  # Create your custom environment in parallels.
    my_callback = MyCallback(configs)  # Create the custom callback.
    Agent = DQN_Agent(config=configs, envs=envs, callback=my_callback)  # Create a DQN agent.

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)


            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()
