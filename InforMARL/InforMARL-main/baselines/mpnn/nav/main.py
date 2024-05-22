import os
import wandb
import socket
from pathlib import Path
import numpy as np
import torch

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))


# import utils
from baselines.mpnn.nav.arguments import get_args
import baselines.mpnn.nav.util as util
from baselines.mpnn.nav.mpnn_runner import Runner
from utils.utils import print_args, connected_to_internet, print_box


np.set_printoptions(suppress=True, precision=4)


def main():
    args = get_args()
    # cuda
    if args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    if args.verbose:
        print_args(args)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / args.scenario_name
        / args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if args.use_wandb:
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

        print("_" * 50)
        print("Creating wandboard...")
        print("_" * 50)
        run = wandb.init(
            config=args,
            project=args.project_name,
            # project=all_args.env_name,
            entity=args.user_name,
            notes=socket.gethostname(),
            name=str(args.algorithm_name)
            + "_"
            + str(args.experiment_name)
            + "_seed"
            + str(args.seed),
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

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env init
    envs = util.make_parallel_envs(args)  # make parallel envs
    num_agents = args.num_agents

    config = {
        "args": args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    if args.verbose:
        print_box("Policy Network", 80)
        print_box(runner.master.policies_list[0], 80)
    runner.run()

    # post process
    envs.close()

    if args.use_wandb:
        run.finish()
    else:
        runner.writer.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writer.close()


if __name__ == "__main__":
    main()
