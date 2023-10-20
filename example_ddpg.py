import os
import argparse
from copy import deepcopy
import numpy as np
import torch.optim
import wandb

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.utils.input_reformat import get_repre_in, get_policy_in
from xuance.torch.agents import get_total_iters


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe.")
    parser.add_argument("--method", type=str, default="ddpg")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def run(args):
    agent_name = args.agent
    set_seed(args.seed)
    # prepare directories for models and log
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)
    args.log_dir = os.path.join(args.log_dir, args.env_id)
    # build environments
    envs = make_envs(args)
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space
    n_envs = envs.num_envs

    # prepare representation
    from xuance.torch.representations import REGISTRY as REGISTRY_Representation
    input_representation = get_repre_in(args)
    representation = REGISTRY_Representation[args.representation](*input_representation)

    # prepare policy
    from xuance.torch.policies import REGISTRY as REGISTRY_Policy
    input_policy = get_policy_in(args, representation)
    policy = REGISTRY_Policy[args.policy](*input_policy)

    # prepare agent
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), args.critic_learning_rate)
    actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                           total_iters=get_total_iters(agent_name, args))
    critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.25,
                                                            total_iters=get_total_iters(agent_name, args))
    from xuance.torch.agents import REGISTRY as REGISTRY_Agent
    agent = REGISTRY_Agent[agent_name](args, envs, policy,
                                       [actor_optimizer, critic_optimizer],
                                       [actor_lr_scheduler, critic_lr_scheduler],
                                       args.device)

    # start running
    envs.reset()
    if not args.test:  # train the model
        n_train_steps = args.running_steps // n_envs
        agent.train(n_train_steps)
        agent.save_model("final_train_model.pth")
        print("Finish training!")
    else:  # test the model
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
    if agent.use_wandb:
        wandb.finish()
    else:
        agent.writer.close()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=None,
                         parser_args=parser)
    run(args)
