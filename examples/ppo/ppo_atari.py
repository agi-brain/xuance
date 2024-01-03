import os
import argparse
from copy import deepcopy
import numpy as np
import torch.optim

from xuance import get_arguments
from xuance.common import space2shape
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.utils import ActivationFunctions


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe.")
    parser.add_argument("--method", type=str, default="ppo")
    parser.add_argument("--env", type=str, default="atari")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--config", type=str, default="./ppo_configs/ppo_atari_config.yaml")

    return parser.parse_args()


def run(args):
    agent_name = args.agent
    set_seed(args.seed)

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)
    args.log_dir = os.path.join(args.log_dir, args.env_id)

    # build environments
    envs = make_envs(args)
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space
    n_envs = envs.num_envs

    # prepare representation
    from xuance.torch.representations import AC_CNN_Atari
    representation = AC_CNN_Atari(input_shape=space2shape(args.observation_space),
                                  kernels=args.kernels,
                                  strides=args.strides,
                                  filters=args.filters,
                                  normalize=None,
                                  initialize=torch.nn.init.orthogonal_,
                                  activation=ActivationFunctions[args.activation],
                                  device=args.device,
                                  fc_hidden_sizes=args.fc_hidden_sizes)

    # prepare policy
    from xuance.torch.policies import Categorical_AC_Policy
    policy = Categorical_AC_Policy(action_space=args.action_space,
                                   representation=representation,
                                   actor_hidden_size=args.actor_hidden_size,
                                   critic_hidden_size=args.critic_hidden_size,
                                   normalize=None,
                                   initialize=torch.nn.init.orthogonal_,
                                   activation=ActivationFunctions[args.activation],
                                   device=args.device)

    # prepare agent
    from xuance.torch.agents import PPOCLIP_Agent, get_total_iters
    optimizer = torch.optim.Adam(policy.parameters(), args.learning_rate, eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                     total_iters=get_total_iters(agent_name, args))
    agent = PPOCLIP_Agent(config=args,
                          envs=envs,
                          policy=policy,
                          optimizer=optimizer,
                          scheduler=lr_scheduler,
                          device=args.device)

    # start running
    envs.reset()
    if args.benchmark:  # run benchmark
        def env_fn():
            args_test = deepcopy(args)
            args_test.parallels = args_test.test_episode
            return make_envs(args_test)

        train_steps = args.running_steps // n_envs
        eval_interval = args.eval_interval // n_envs
        test_episode = args.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = agent.test(env_fn, test_episode)
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            agent.train(eval_interval)
            test_scores = agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": agent.current_step}
                # save best model
                agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if not args.test:  # train the model without testing
            n_train_steps = args.running_steps // n_envs
            agent.train(n_train_steps)
            agent.save_model("final_train_model.pth")
            print("Finish training!")
        else:  # test a trained model
            def env_fn():
                args_test = deepcopy(args)
                args_test.parallels = 1
                return make_envs(args_test)

            agent.render = True
            agent.load_model(agent.model_dir_load, args.seed)
            scores = agent.test(env_fn, args.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")

    # the end.
    envs.close()
    agent.finish()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)
