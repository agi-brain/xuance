import argparse
from distutils.util import strtobool


def get_config():
    parser = argparse.ArgumentParser(
        description="OFF-POLICY", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="rmatd3",
        choices=[
            "rmatd3",
            "rmaddpg",
            "rmasac",
            "qmix",
            "vdn",
            "matd3",
            "maddpg",
            "masac",
            "mqmix",
            "mvdn",
        ],
    )  # masac not implemented;
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
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    parser.add_argument("--cuda", action="store_false", default=True)
    parser.add_argument("--cuda_deterministic", action="store_false", default=True)
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for training rollout",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollout",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=2000000,
        help="Number of env steps to train for",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date "
        "to wandb server. or else will use tensorboard to log data.",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for "
        "simply collecting training data.",
    )

    # env parameters
    parser.add_argument("--env_name", type=str, default="StarCraft2")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The world size of MPE; it will range from "
        "-world_size/2 to world_size/2",
    )
    parser.add_argument(
        "--num_scripted_agents",
        type=int,
        default=0,
        help="The number of scripted agents in MPE",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        choices=["local", "global", "nbd"],
        default="global",
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
        "--use_comm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use communication " "channel for agent observation",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=25, help="Max length for any episode"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5000,
        help="Max # of transitions that replay buffer can contain",
    )
    parser.add_argument(
        "--use_reward_normalization",
        action="store_true",
        default=False,
        help="Whether to normalize rewards in replay buffer",
    )
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="Whether to use popart to normalize the target loss",
    )
    parser.add_argument(
        "--popart_update_interval_step",
        type=int,
        default=2,
        help="After how many train steps popart should be updated",
    )

    # prioritized experience replay
    parser.add_argument(
        "--use_per",
        action="store_true",
        default=False,
        help="Whether to use prioritized experience replay",
    )
    parser.add_argument(
        "--per_nu",
        type=float,
        default=0.9,
        help="Weight of max TD error in formation of PER weights",
    )
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="Alpha term for prioritized experience replay",
    )
    parser.add_argument(
        "--per_eps",
        type=float,
        default=1e-6,
        help="Eps term for prioritized experience replay",
    )
    parser.add_argument(
        "--per_beta_start",
        type=float,
        default=0.4,
        help="Starting beta term for prioritized experience replay",
    )

    # network parameters
    parser.add_argument(
        "--use_centralized_Q",
        action="store_false",
        default=True,
        help="Whether to use centralized Q function",
    )
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agents share the same policy",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU", action="store_false", default=True, help="Whether to use ReLU"
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain", type=float, default=0.01, help="The gain # of last action layer"
    )
    parser.add_argument(
        "--use_conv1d", action="store_true", default=False, help="Whether to use conv1d"
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )

    # recurrent parameters
    parser.add_argument(
        "--prev_act_inp",
        action="store_true",
        default=False,
        help="Whether the actor input takes in previous actions as part of its input",
    )
    parser.add_argument(
        "--use_rnn_layer",
        action="store_false",
        default=True,
        help="Whether to use a recurrent policy",
    )
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_false",
        default=True,
        help="Whether to use a naive recurrent policy",
    )
    # TODO now only 1 is support
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=80,  # NOTE: check if this should be 10 as onpolicy
        help="Time length of chunks used to train via BPTT",
    )
    parser.add_argument(
        "--burn_in_time",
        type=int,
        default=0,
        help="Length of burn in time for RNN training, see R2D2 paper",
    )

    # attn parameters
    parser.add_argument("--attn", action="store_true", default=False)
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool", action="store_false", default=True)
    parser.add_argument("--use_cat_self", action="store_false", default=True)

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Adam")
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # algo common parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of buffer transitions to train on at once",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for env"
    )
    parser.add_argument("--use_max_grad_norm", action="store_false", default=True)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_true",
        default=False,
        help="Whether to use Huber loss for critic update",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # soft update parameters
    parser.add_argument(
        "--use_soft_update",
        action="store_false",
        default=True,
        help="Whether to use soft update",
    )
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak update rate")
    # hard update parameters
    parser.add_argument(
        "--hard_update_interval_episode",
        type=int,
        default=200,
        help="After how many episodes the lagging target should be updated",
    )
    parser.add_argument(
        "--hard_update_interval",
        type=int,
        default=200,
        help="After how many timesteps the lagging target should be updated",
    )
    # rmatd3 parameters
    parser.add_argument(
        "--target_action_noise_std",
        default=0.2,
        help="Target action smoothing noise for matd3",
    )
    # rmasac parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Initial temperature")
    parser.add_argument(
        "--target_entropy_coef", type=float, default=0.5, help="Initial temperature"
    )
    parser.add_argument(
        "--automatic_entropy_tune",
        action="store_false",
        default=True,
        help="Whether use a centralized critic",
    )
    # qmix parameters
    parser.add_argument(
        "--use_double_q",
        action="store_false",
        default=True,
        help="Whether to use double q learning",
    )
    parser.add_argument(
        "--hypernet_layers",
        type=int,
        default=2,
        help="Number of layers for hypernetworks. Must be either 1 or 2",
    )
    parser.add_argument(
        "--mixer_hidden_dim",
        type=int,
        default=32,
        help="Dimension of hidden layer of mixing network",
    )
    parser.add_argument(
        "--hypernet_hidden_dim",
        type=int,
        default=64,
        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2",
    )

    # exploration parameters
    parser.add_argument(
        "--num_random_episodes",
        type=int,
        default=5,
        help="Number of episodes to add to buffer with purely random actions",
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=1.0,
        help="Starting value for epsilon, for eps-greedy exploration",
    )
    parser.add_argument(
        "--epsilon_finish",
        type=float,
        default=0.05,
        help="Ending value for epsilon, for eps-greedy exploration",
    )
    parser.add_argument(
        "--epsilon_anneal_time",
        type=int,
        default=50000,
        help="Number of episodes until epsilon reaches epsilon_finish",
    )
    parser.add_argument("--act_noise_std", type=float, default=0.1, help="Action noise")

    # train parameters
    parser.add_argument(
        "--actor_train_interval_step",
        type=int,
        default=2,
        help="After how many critic updates actor should be updated",
    )
    parser.add_argument(
        "--train_interval_episode",
        type=int,
        default=1,
        help="Number of env steps between updates to actor/critic",
    )
    parser.add_argument(
        "--train_interval",
        type=int,
        default=100,
        help="Number of episodes between updates to actor/critic",
    )
    parser.add_argument("--use_value_active_masks", action="store_true", default=False)

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_false",
        default=True,
        help="Whether to conduct the evaluation",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10000,
        help="After how many episodes the policy should be evaled",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=32,
        help="How many episodes to collect for each eval",
    )

    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100000,
        help="After how many episodes of training the policy model should be saved",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="After how many episodes of training the policy model should be saved",
    )

    # pretained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser
