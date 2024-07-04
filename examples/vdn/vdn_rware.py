import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import VDN_Agents


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe.")
    parser.add_argument("--env-id", type=str, default="rware-tiny-2ag-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir=f"vdn_rware_configs/{parser.env_id}.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)  # Set the random seed.
    envs = make_envs(configs)  # Make the environment.
    Agents = VDN_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.

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
