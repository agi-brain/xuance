import argparse
from distutils.util import strtobool


def get_args():
    parser = argparse.ArgumentParser(description="MPNN")

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="mpnn")
    parser.add_argument(
        "--project_name",
        type=str,
        default="test",
        help="project name to store logs under",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; " "or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. "
        "if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for "
        "simply collecting training data.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date "
        "to wandb server. or else will use tensorboard to log data.",
    )

    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default="MPE",
        choices=["MPE", "GraphMPE"],
        help="specify the name of environment",
    )
    parser.add_argument("--scenario_name", default="navigation")
    parser.add_argument(
        "--world_size",
        type=float,
        default=2,
        help="The world size of MPE will range from \pm world_size/2",
    )
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument(
        "--num_scripted_agents",
        type=int,
        default=0,
        help="The number of scripted agents in MPE",
    )
    # NOTE: important to use `local` obs type because original paper uses this
    parser.add_argument(
        "--obs_type",
        type=str,
        choices=["local", "global", "nbd"],
        default="local",
        help="Whether to use local obs for navigation.py",
    )
    parser.add_argument(
        "--max_edge_dist",
        type=float,
        default=1,
        help="Maximum distance above which edges cannot be "
        "connected between the entities; used for `obs_type==ndb_obs`",
    )
    parser.add_argument(
        "--num_nbd_entities",
        type=int,
        default=3,
        help="Number of entities to be considered as neighbors "
        "for `obs_type==ndb_obs`",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )
    parser.add_argument(
        "--use_comm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use communication " "channel for agent observation",
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
        "--masking",
        action="store_true",
        help="restrict communication to within some threshold",
    )
    parser.add_argument(
        "--mask-dist", type=float, default=1.0, help="distance to restrict comms"
    )
    parser.add_argument(
        "--dropout-masking", action="store_true", help="dropout masking enabled"
    )
    parser.add_argument(
        "--entity-mp",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="enable entity message passing",
    )
    parser.add_argument(
        "--identity-size", default=0, type=int, help="size of identity vector"
    )

    # training
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="number of forward steps in PPO (default: 128)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=int(50e6),
        help="number of frames to train (default: 50e6)",
    )
    parser.add_argument("--arena-size", type=int, default=1, help="size of arena")

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )
    # evaluation
    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, "
        "start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )
    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. "
        "If set, start render. Note: something, the environment "
        "has internal render process which is not controlled by "
        "this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )
    # parser.add_argument('--num-eval-episodes', type=int, default=30, help='number of episodes to evaluate with')
    # parser.add_argument('--dist-threshold', type=float, default=0.1, help='distance within landmark is considered covered (for simple_spread)')
    # parser.add_argument('--render', action='store_true')
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="record evaluation video",
    )

    # PPO
    parser.add_argument(
        "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.95, help="gae parameter (default: 0.95)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )

    # logging
    parser.add_argument(
        "--save-dir", default="tmp", help="directory to save models (default: tmp)"
    )
    parser.add_argument("--log-dir", default="logs", help="directory to save logs")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="save interval, one save per n updates (default: 200)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )

    # Miscellaneous
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--load-dir", default=None, help="filename to load all policies from"
    )
    parser.add_argument("--eval-interval", default=50, type=int)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_false",
        default=True,
        help="by default, print args and network at the begining.",
    )

    # we always set these to TRUE, so automating this
    parser.add_argument("--no-clipped-value-loss", action="store_true")

    args = parser.parse_args()

    args.clipped_value_loss = not args.no_clipped_value_loss

    # if args.continue_training:
    #     assert args.load_dir is not None and os.path.exists(args.load_dir), \
    #     "Please specify valid model file to load if you want to continue training"

    if args.identity_size > 0:
        assert (
            args.identity_size >= args.num_agents
        ), "identity size should either be 0 or >= number of agents!"

    if not args.masking:
        args.mask_dist = None
    elif args.masking and args.dropout_masking:
        args.mask_dist = -10
    return args
