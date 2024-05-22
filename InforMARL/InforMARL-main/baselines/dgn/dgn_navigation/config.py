import argparse
from distutils.util import strtobool


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private
    hyperparameters only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including
            `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for
            some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting
            training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server.
            or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or
            else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network;
            set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode;
            or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal init for weights and 0 init for biases.
            or else, will use xavier uniform init.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)


    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation.
            If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="dgn")

    parser.add_argument(
        "--project_name",
        type=str,
        default="test",
        help="project name to store logs under",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dgn",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
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
        "--env_name", type=str, default="MPE", help="specify the name of environment"
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="navigation_dgn",
        help="Which scenario to run on",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dgn_orig",
        help="Which model to run on. " "the other option is dgn_atoc",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument(
        "--world_size",
        type=float,
        default=2,
        help="The world size of MPE; it will range from "
        "-world_size/2 to world_size/2",
    )
    parser.add_argument(
        "--num_scripted_agents",
        type=int,
        default=0,
        help="number of non-controllable players",
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
    parser.add_argument(
        "--max_edge_dist",
        type=float,
        default=1,
        help="Maximum distance above which edges cannot be "
        "connected between the entities",
    )
    parser.add_argument(
        "--graph_feat_type",
        type=str,
        default="global",
        choices=["global", "relative"],
        help="Whether to use " "'global' node/edge feats or 'relative'",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="The degree of connections in graphs for each node",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=25, help="Max length for any episode"
    )

    # network parameters
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for "
        "weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain", type=float, default=0.01, help="The gain # of last action layer"
    )

    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )

    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
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

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    # misc parameters
    parser.add_argument(
        "--verbose",
        action="store_false",
        default=True,
        help="by default, print args and network at the begining.",
    )

    # DGN parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.1,
        help="threshold for setting adjacency matrix entry to 1",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="hidden dimension for DGN"
    )
    parser.add_argument("--capacity", type=int, default=65000, help="buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--n_epoch", type=int, default=25, help="number of epoch for each episode"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.9, help="epsilon value for exploration"
    )
    parser.add_argument(
        "--comm_flag", type=int, default=1, help="Flag to enable graph communication"
    )  # TODO what's this????
    parser.add_argument(
        "--tau", type=float, default=0.98, help="tau value for soft update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Factor")

    # max_step = 500
    # i_episode = 0

    return parser
