import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_ENV

from xuance.torch.utils.operations import set_seed
from td3_bc_agent import TD3_BC_Agent
from walker2d import Walker2d
from offline_utils import load_d4rl_dataset

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: TD3_BC for MuJoCo.")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="td3_bc_configs/td3_bc_walker2d.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_ENV[configs.env_name] = Walker2d

    set_seed(configs.seed)  # Set the random seed.

    dataset, state_mean, state_std = load_d4rl_dataset(
        dataset_name=configs.dataset,
        max_episode_steps=configs.test_steps // configs.test_episode,
        obsnorm=configs.normalize_obs_offline,
        rewnorm=configs.normalize_reward_offline
    )
    configs.state_mean = state_mean
    configs.state_std = state_std

    envs = make_envs(configs)  # Make the environment.
    Agent = TD3_BC_Agent(config=configs, envs=envs)  # Create the TD3_BC agent.
    Agent.load_dataset(dataset=dataset)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id,
                         "Seed":configs.env_seed}
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

        best_scores_info = {"mean": -np.inf,
                            "std": -np.inf,
                            "step": -np.inf}
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
            print(f"Normalized-Test-Episode-Rewards: {test_scores}")
            print(f"D4RL-Score: %.3f" % np.mean(test_scores))
        # end benchmarking
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
            #Agent.train(configs.running_steps // configs.parallels)
            Agent.train(configs.running_step)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()
