from distutils.util import strtobool
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

import numpy as np
from pathlib import Path
import socket
import wandb
import setproctitle
import torch

from utils.utils import print_args, print_box, connected_to_internet
from baselines.offpolicy.config import get_config
from baselines.offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from multiagent.MPE_env import MPEEnv
from baselines.offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents")
    parser.add_argument(
        "--use_same_share_obs",
        action="store_false",
        default=True,
        help="Whether to use available actions",
    )
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
        default=0.05,
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


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    print_args(all_args)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        # init wandb
        print("_" * 50)
        print("Creating wandboard...")
        print("_" * 50)
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
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

    # create env
    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
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
        from baselines.offpolicy.runner.rnn.mpe_runner import MPERunner as Runner

        assert (
            all_args.n_rollout_threads == 1
        ), "only support 1 env in recurrent version."
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from baselines.offpolicy.runner.mlp.mpe_runner import MPERunner as Runner

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
        "use_same_share_obs": all_args.use_same_share_obs,
        "run_dir": run_dir,
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
