from xuance.common import space2shape
from copy import deepcopy
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions, InitializeFunctions
from xuance.torch.policies import Policy_Inputs, Policy_Inputs_All
from xuance.torch.representations import Representation_Inputs, Representation_Inputs_All
from operator import itemgetter
from gymnasium.spaces import Box, Discrete
import torch


def get_repre_in(args, name=None):
    representation_name = args.representation if name is None else name
    input_dict = deepcopy(Representation_Inputs_All)
    if args.env_name in ["StarCraft2", "Football", "MAgent2"]:
        input_dict["input_shape"] = (args.dim_obs, )
    elif isinstance(args.observation_space, dict):
        input_dict["input_shape"] = space2shape(args.observation_space[args.agent_keys[0]])
    else:
        input_dict["input_shape"] = space2shape(args.observation_space)

    if representation_name in ["Basic_MLP", "CoG_MLP"]:
        input_dict["hidden_sizes"] = args.representation_hidden_size
    elif representation_name in ["Basic_RNN"]:
        input_dict["hidden_sizes"] = {
            "fc_hidden_sizes": args.fc_hidden_sizes,
            "recurrent_hidden_size": args.recurrent_hidden_size
        }
    else:
        if representation_name in ["Basic_CNN", "CoG_CNN", "AC_CNN_Atari"]:
            input_dict["kernels"] = args.kernels
            input_dict["strides"] = args.strides
            input_dict["filters"] = args.filters
        if representation_name in ["AC_CNN_Atari"]:
            input_dict["fc_hidden_sizes"] = args.fc_hidden_sizes

    input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
    input_dict["initialize"] = torch.nn.init.orthogonal_
    input_dict["activation"] = ActivationFunctions[args.activation]
    input_dict["device"] = args.device

    input_list = itemgetter(*Representation_Inputs[representation_name])(input_dict)

    return list(input_list)


def get_policy_in(args, representation):
    policy_name = args.policy
    input_dict = deepcopy(Policy_Inputs_All)
    input_dict["action_space"] = args.action_space
    input_dict["representation"] = representation
    if policy_name in ["Basic_Q_network", "Duel_Q_network", "Noisy_Q_network", "C51_Q_network", "QR_Q_network"]:
        input_dict["hidden_sizes"] = args.q_hidden_size
        if policy_name == "C51_Q_network":
            input_dict['vmin'] = args.vmin
            input_dict['vmax'] = args.vmax
            input_dict['atom_num'] = args.atom_num
        elif policy_name == "QR_Q_network":
            input_dict['quantile_num'] = args.quantile_num
    elif policy_name in ['PDQN_Policy', 'MPDQN_Policy', 'SPDQN_Policy']:
        input_dict['observation_space'] = args.observation_space
        input_dict['conactor_hidden_size'] = args.conactor_hidden_size
        input_dict['qnetwork_hidden_size'] = args.qnetwork_hidden_size
    elif policy_name in ['DRQN_Policy']:
        input_dict["rnn"] = args.rnn
        input_dict["recurrent_hidden_size"] = args.recurrent_hidden_size
        input_dict["recurrent_layer_N"] = args.recurrent_layer_N
        input_dict["dropout"] = args.dropout
    else:
        input_dict["actor_hidden_size"] = args.actor_hidden_size
        if policy_name in ["Categorical_AC", "Categorical_PPG", "Gaussian_AC", "Discrete_SAC", "Gaussian_SAC", "Gaussian_PPG", "DDPG_Policy", "TD3_Policy"]:
            input_dict["critic_hidden_size"] = args.critic_hidden_size
    input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
    input_dict["initialize"] = torch.nn.init.orthogonal_
    input_dict["activation"] = ActivationFunctions[args.activation]
    try: input_dict["activation_action"] = ActivationFunctions[args.activation_action]
    except: pass
    input_dict["device"] = args.device

    if policy_name == "Gaussian_Actor":
        input_dict["fixed_std"] = None
    if policy_name == "DRQN_Policy":
        return input_dict
    input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
    return list(input_list)


def get_policy_in_marl(args, representation, mixer=None, ff_mixer=None, qtran_mixer=None):
    policy_name = args.policy
    input_dict = deepcopy(Policy_Inputs_All)
    try: input_dict["state_dim"] = args.dim_state[0]
    except: pass

    if isinstance(args.action_space, dict):
        input_dict["action_space"] = args.action_space[args.agent_keys[0]]
    else:
        input_dict["action_space"] = args.action_space

    try: input_dict["n_agents"] = args.n_agents
    except: pass
    input_dict["representation"] = representation
    input_dict["mixer"] = mixer
    input_dict["ff_mixer"] = ff_mixer
    input_dict["qtran_mixer"] = qtran_mixer
    if policy_name in ["Basic_Q_network_marl", "Mixing_Q_network", "Weighted_Mixing_Q_network",
                       "Qtran_Mixing_Q_network", "MF_Q_network"]:
        input_dict["hidden_sizes"] = args.q_hidden_size
    else:
        input_dict["actor_hidden_size"] = args.actor_hidden_size
        try: input_dict["critic_hidden_size"] = args.critic_hidden_size
        except: pass

    input_dict["initialize"] = InitializeFunctions[args.initialize] if hasattr(args, "initialize") else None
    input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
    input_dict["activation"] = ActivationFunctions[args.activation]
    try: input_dict["activation_action"] = ActivationFunctions[args.activation_action]
    except: pass

    input_dict["device"] = args.device
    if policy_name == "Gaussian_Actor":
        input_dict["fixed_std"] = None
    input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
    return list(input_list)
