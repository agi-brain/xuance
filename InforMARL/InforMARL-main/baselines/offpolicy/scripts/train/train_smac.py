import sys
import os
import numpy as np
from pathlib import Path
import wandb
import socket
import setproctitle
import torch
from baselines.offpolicy.config import get_config
from baselines.offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from baselines.offpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from baselines.offpolicy.envs.starcraft2.smac_maps import get_map_params
from baselines.offpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    parser.add_argument(
        "--map_name", type=str, default="3m", help="Which smac map to run on"
    )
    parser.add_argument(
        "--use_available_actions",
        action="store_false",
        default=True,
        help="Whether to use available actions",
    )
    parser.add_argument(
        "--use_same_share_obs",
        action="store_false",
        default=True,
        help="Whether to use available actions",
    )
    parser.add_argument(
        "--use_global_all_local_state",
        action="store_true",
        default=False,
        help="Whether to use available actions",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.map_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = make_train_env(all_args)
    buffer_length = get_map_params(all_args.map_name)["limit"]
    print(buffer_length)
    num_agents = get_map_params(all_args.map_name)["n_agents"]

    # create policies and mapping fn
    if all_args.share_policy:
        print(env.share_observation_space[0])
        policy_info = {
            "policy_0": {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[0],
                "share_obs_space": env.share_observation_space[0],
                "act_space": env.action_space[0],
            }
        }

        def policy_mapping_fn(id):
            return "policy_0"

    else:
        policy_info = {
            "policy_"
            + str(agent_id): {
                "cent_obs_dim": get_dim_from_space(
                    env.share_observation_space[agent_id]
                ),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[agent_id],
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id],
            }
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id):
            return "policy_" + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        from offpolicy.runner.rnn.smac_runner import SMACRunner as Runner

        assert (
            all_args.n_rollout_threads == 1
        ), "only support 1 env in recurrent version."
        eval_env = make_train_env(all_args)
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.smac_runner import SMACRunner as Runner

        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    config = {
        "args": all_args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "eval_env": eval_env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "buffer_length": buffer_length,
        "use_same_share_obs": all_args.use_same_share_obs,
        "use_available_actions": all_args.use_available_actions,
    }

    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
