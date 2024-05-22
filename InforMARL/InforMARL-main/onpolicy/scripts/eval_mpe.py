#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import sys
import os
from typing import Dict
import wandb
import numpy as np
from pathlib import Path
import torch

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)


def make_render_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.1,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def modify_args(
    model_dir: str,
    args: argparse.Namespace,
    exclude_args: list = [
        "model_dir",
        "num_agents",
        "num_obstacles",
        "num_landmarks",
        "render_episodes",
        "world_size",
        "seed",
    ],
):
    """
    Modify the args used to train the model
    """
    import yaml

    with open(str(model_dir) + "/config.yaml") as f:
        ydict = yaml.load(f)

    print("_" * 50)
    for k, v in ydict.items():
        if k in exclude_args:
            print(f"Using {k} = {vars(args)[k]}")
            # print(f"Skipping {k} with value {args.k}")
            continue
        # all args have 'values' and 'desc' as keys
        if type(v) == dict:
            if "value" in v.keys():
                # print(f'Setting attr {k} to {ydict[k]["value"]}')
                setattr(args, k, ydict[k]["value"])
    print("_" * 50)

    # set some args manually
    args.cuda = False
    args.use_wandb = False
    args.use_render = True
    args.save_gifs = True
    args.n_rollout_threads = 1

    return args


def main(args):
    # model_dir = 'trained_models/navigation/Navigation/rmappo/wandb/offline-run-20210720_220614-1eqhk4l1/files'
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args = modify_args(all_args.model_dir, all_args)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (
            all_args.use_recurrent_policy and all_args.use_naive_recurrent_policy
        ) == False, "check recurrent policy!"
    else:
        raise NotImplementedError

    assert all_args.use_render, "Need to set use_render be True"
    assert not (
        all_args.model_dir == None or all_args.model_dir == ""
    ), "set model_dir first"
    assert all_args.n_rollout_threads == 1, "only support to use 1 env to render."

    device = torch.device("cpu")

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    run_dir = None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    # print_args(config['all_args'])

    runner = Runner(config)
    # actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    # runner.policy.actor.load_state_dict(actor_state_dict)
    runner.render(True)

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
