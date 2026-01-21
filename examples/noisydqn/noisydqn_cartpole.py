import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import NoisyDQN_Agent


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: NoisyDQN for CartPole.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="noisydqn_configs/noisydqn_cartpole.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = NoisyDQN_Agent(config=configs, envs=envs)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        configs_test = deepcopy(configs)
        configs_test.parallels = configs_test.test_episode
        test_envs = make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        test_envs.close()
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            configs.parallels = configs.test_episode
            test_envs = make_envs(configs)
            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(test_episodes=configs.test_episode, test_envs=test_envs, close_envs=True)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()
