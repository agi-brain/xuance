import argparse

import os

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe.")
    parser.add_argument("--method", type=str, default="sac")
    parser.add_argument("--env", type=str, default="Box2D")
    parser.add_argument("--env-id", type=str, default="LunarLander-v2")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--config", type=str, default="./configs/sac.yaml")

    return parser.parse_known_args()[0]


import os
from copy import deepcopy
import numpy as np
import torch.optim

from xuance.common import space2shape
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.utils import ActivationFunctions


def run(args):
    agent_name = args.agent  # get the name of Agent.
    set_seed(args.seed)  # set random seed.

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)  # the path for saved model.
    args.log_dir = os.path.join(args.log_dir, args.env_id)  # the path for logger file.

    # build environments
    envs = make_envs(args)  # create simulation environments
    args.observation_space = envs.observation_space  # get observation space
    args.action_space = envs.action_space  # get action space
    n_envs = envs.num_envs  # get the number of vectorized environments.

    # prepare representation
    from xuance.torch.representations import Basic_MLP
    representation = Basic_MLP(input_shape=space2shape(args.observation_space),
                               hidden_sizes=args.representation_hidden_size,
                               normalize=None,
                               initialize=torch.nn.init.orthogonal_,
                               activation=ActivationFunctions[args.activation],
                               device=args.device)  # create representation

    # prepare policy
    from xuance.torch.policies import  C51Qnetwork,Categorical_SAC_Policy
    policy = Categorical_SAC_Policy(
        action_space=args.action_space,

        representation=representation,
        actor_hidden_size=args.actor_hidden_size,
        critic_hidden_size=args.critic_hidden_size,
        normalize=None,
        initialize=torch.nn.init.orthogonal_,
        activation=ActivationFunctions[args.activation],
        device=args.device)  # create Gaussian policy

    # prepare agent
    from xuance.torch.agents import SACDIS_Agent, get_total_iters, C51_Agent
    # optimizer = torch.optim.Adam(policy.parameters(), args.learning_rate, eps=1e-5)  # create optimizer
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), args.critic_learning_rate)
    actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                     total_iters=get_total_iters(agent_name,
                                                                               args))  # for learning rate decay
    critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.25,
                                                     total_iters=get_total_iters(agent_name,
                                                                                 args))  # for learning rate decay

    agent = SACDIS_Agent(

        config=args,
        envs=envs,
        policy=policy,
        optimizer=[actor_optimizer,critic_optimizer],
        scheduler=[actor_lr_scheduler,critic_lr_scheduler],
        device=args.device)  # create a sac agent

    # start running
    envs.reset()  # reset the environments
    if args.benchmark:  # run benchmark
        def env_fn():  # for creating testing environments
            args_test = deepcopy(args)
            args_test.parallels = args_test.test_episode  # set number of testing environments.
            return make_envs(args_test)  # make testing environments.

        train_steps = args.running_steps // n_envs  # calculate the total running steps.
        eval_interval = args.eval_interval // n_envs  # calculate the number of training steps per epoch.
        test_episode = args.test_episode  # calculate the number of testing episodes.
        num_epoch = int(train_steps / eval_interval)  # calculate the number of epochs.

        test_scores = agent.test(env_fn, test_episode)  # first test
        best_scores_info = {"mean": np.mean(test_scores),  # average episode scores.
                            "std": np.std(test_scores),  # the standard deviation of the episode scores.
                            "step": agent.current_step}  # current step
        # print(f"mean: {best_scores_info['mean']:.2f}, std={best_scores_info['std']:.2f}")
        for i_epoch in range(num_epoch):  # begin benchmarking
            print(f"Epoch: {i_epoch+1}/{num_epoch}:")
            agent.train(eval_interval)  # train the model for some steps
            test_scores = agent.test(env_fn, test_episode)  # test the model for some episodes
            print(f"mean: {best_scores_info['mean']:.2f}, std={best_scores_info['std']:.2f}")
            if np.mean(test_scores) > best_scores_info["mean"]:  # if current score is better than history
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": agent.current_step}
                # save best model
                agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        # 如果test>0，测试模型
        if args.test:  # train the model without testing
            def env_fn():  # for creating testing environments
                args_test = deepcopy(args)
                args_test.parallels = 1
                return make_envs(args_test)

            agent.render = True
            print(agent.model_dir_load)
            # print("files:",os.listdir(agent.model_dir_load))
            agent.load_model(agent.model_dir_load)  # load the model file
            scores = agent.test(env_fn, args.test_episode)  # test the model
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:  # test a trained model
            # n_train_steps = args.running_steps // n_envs  # calculate the total steps of training
            # agent.train(n_train_steps)  # train the model directly.
            # agent.save_model("final_train_model.pth")  # save the final model file.
            # print("Finish training!")
            pass
    # the end.

    envs.close()  # close the environment
    agent.finish()  # finish the example


from xuance import get_arguments
from xuance import get_runner

if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)
    # print(args)
