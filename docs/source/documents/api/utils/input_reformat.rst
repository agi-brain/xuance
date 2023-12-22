Input Reformation
=================================


This module defines functions related to configuring and obtaining input specifications for reinforcement learning policies and representations.

.. raw:: html

    <br><hr>

PyTorch
--------------------------------------------------

.. py:function::
  xuance.torch.utils.input_reformat.get_repre_in(args, name)

  This function obtains input specifications for representations.

  :param args: arguments.
  :type args: Namespace
  :param name: the name of the representations.
  :type name: str
  :return: a list of the input variables for representation module.
  :rtype: list

.. py:function::
  xuance.torch.utils.input_reformat.get_policy_in(args, representation)

  This function obtains input specifications for policies.

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: nn.Module
  :return: a list of input specifications.
  :rtype: list

.. py:function::
  xuance.torch.utils.input_reformat.get_policy_in_marl(args, representation, mixer, ff_mixer, qtran_mixer)

  This function is similar to get_policy_in, but it is designed for multi-agent reinforcement learning (MARL) scenarios.
  It takes additional mixer-related parameters.

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: nn.Module
  :param mixer: The mixer for independent values.
  :type mixer: nn.Module
  :param ff_mixer: the feed forward mixer, default is None.
  :type ff_mixer: nn.Module
  :param qtran_mixer: the QTRAN mixer, default is None.
  :type qtran_mixer: nn.Module
  :return: a list of input specifications.
  :rtype: list

.. raw:: html

    <br><hr>

TensorFlow
-------------------------------------------------

.. py:function::
  xuance.tensorflow.utils.input_reformat.get_repre_in(args)

  This function obtains input specifications for representations.

  :param args: the arguments.
  :type args: Namespace
  :return: a list of the input variables for representation module.
  :rtype: list

.. py:function::
  xuance.tensorflow.utils.input_reformat.get_policy_in(args, representation)

  This function obtains input specifications for policies.

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: tk.Model
  :return: a list of input specifications.
  :rtype: list

.. py:function::
  xuance.tensorflow.utils.input_reformat.get_policy_in_marl(args, representation, mixer, ff_mixer, qtran_mixer)

  This function is similar to get_policy_in, but it is designed for multi-agent reinforcement learning (MARL) scenarios. 
  It takes additional mixer-related parameters.

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: tk.Model
  :param mixer: The mixer for independent values.
  :type mixer: tk.Model
  :param ff_mixer: the feed forward mixer, default is None.
  :type ff_mixer: tk.Model
  :param qtran_mixer: the QTRAN mixer, default is None.
  :type qtran_mixer: tk.Model
  :return: a list of input specifications.
  :rtype: list

.. raw:: html

    <br><hr>

MindSpore
---------------------------------------------------

.. py:function::
  xuance.mindspore.utils.input_reformat.get_repre_in(args)

  This function obtains input specifications for representations.

  :param args: the arguments.
  :type args: Namespace
  :return: a list of the input variables for representation module.
  :rtype: list

.. py:function::
  xuance.mindspore.utils.input_reformat.get_policy_in(args, representation)

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: nn.Cell
  :return: a list of the input variables for representation module.
  :rtype: list

.. py:function::
  xuance.mindspore.utils.input_reformat.get_policy_in_marl(args, representation, mixer, ff_mixer, qtran_mixer)

  This function is similar to get_policy_in, but it is designed for multi-agent reinforcement learning (MARL) scenarios. 
  It takes additional mixer-related parameters.

  :param args: the arguments.
  :type args: Namespace
  :param representation: The representation module.
  :type representation: nn.Cell
  :param mixer: The mixer for independent values.
  :type mixer: nn.Cell
  :param ff_mixer: the feed forward mixer, default is None.
  :type ff_mixer: nn.Cell
  :param qtran_mixer: the QTRAN mixer, default is None.
  :type qtran_mixer: nn.Cell
  :return: a list of input specifications.
  :rtype: list

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from xuance.common import space2shape

        from copy import deepcopy
        from xuance.torch.utils import ActivationFunctions, NormalizeFunctions, InitializeFunctions
        from xuance.torch.policies import Policy_Inputs, Policy_Inputs_All
        from xuance.torch.representations import Representation_Inputs, Representation_Inputs_All
        from operator import itemgetter
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
            except: input_dict["state_dim"] = None

            if args.env_name in ["StarCraft2", "Football"]:
                input_dict["action_space"] = args.action_space
            else:
                input_dict["action_space"] = args.action_space[args.agent_keys[0]]

            try: input_dict["n_agents"] = args.n_agents
            except: input_dict["n_agents"] = 1
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
                except: input_dict["critic_hidden_size"] = None

            input_dict["initialize"] = InitializeFunctions[args.initialize] if hasattr(args, "initialize") else None
            input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
            input_dict["activation"] = ActivationFunctions[args.activation]

            input_dict["device"] = args.device
            if policy_name == "Gaussian_Actor":
                input_dict["fixed_std"] = None
            input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
            return list(input_list)

  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.common import space2shape
        from copy import deepcopy
        from xuance.torch.utils import ActivationFunctions, NormalizeFunctions, InitializeFunctions
        from xuance.tensorflow.policies import Policy_Inputs, Policy_Inputs_All
        from xuance.tensorflow.representations import Representation_Inputs, Representation_Inputs_All
        from operator import itemgetter
        import tensorflow.keras as tk


        def get_repre_in(args):
            representation_name = args.representation
            input_dict = deepcopy(Representation_Inputs_All)
            if isinstance(args.observation_space, dict):
                input_dict["input_shape"] = space2shape(args.observation_space[args.agent_keys[0]])
            else:
                input_dict["input_shape"] = space2shape(args.observation_space)

            if representation_name in ["Basic_MLP", "CoG_MLP"]:
                input_dict["hidden_sizes"] = args.representation_hidden_size
            else:
                if representation_name in ["Basic_CNN", "CoG_CNN"]:
                    input_dict["kernels"] = args.kernels
                    input_dict["strides"] = args.strides
                    input_dict["filters"] = args.filters

            input_dict["normalize"] = None
            input_dict["initialize"] = tk.initializers.GlorotUniform(seed=0)
            input_dict["activation"] = tk.layers.Activation('relu')
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
            input_dict["normalize"] = None
            input_dict["initialize"] = tk.initializers.GlorotUniform(seed=0)
            input_dict["activation"] = tk.layers.Activation('relu')
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
            except: input_dict["state_dim"] = None

            if args.env_name in ["StarCraft2", "Football"]:
                input_dict["action_space"] = args.action_space
            else:
                input_dict["action_space"] = args.action_space[args.agent_keys[0]]

            try: input_dict["n_agents"] = args.n_agents
            except: input_dict["n_agents"] = 1
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
                except: input_dict["critic_hidden_size"] = None

            # input_dict["initialize"] = InitializeFunctions[args.initialize] if hasattr(args, "initialize") else None
            # input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
            # input_dict["activation"] = ActivationFunctions[args.activation]

            input_dict["normalize"] = None
            input_dict["initialize"] = None
            input_dict["activation"] = tk.layers.Activation('relu')

            input_dict["device"] = args.device
            if policy_name == "Gaussian_Actor":
                input_dict["fixed_std"] = None
            input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
            return list(input_list)


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.common import space2shape
        from copy import deepcopy
        from xuance.mindspore.utils import ActivationFunctions, NormalizeFunctions, InitializeFunctions
        from xuance.mindspore.policies import Policy_Inputs, Policy_Inputs_All
        from xuance.mindspore.representations import Representation_Inputs, Representation_Inputs_All
        from operator import itemgetter
        import mindspore.nn as nn
        from mindspore.common.initializer import TruncatedNormal


        def get_repre_in(args):
            representation_name = args.representation
            input_dict = deepcopy(Representation_Inputs_All)
            if isinstance(args.observation_space, dict):
                input_dict["input_shape"] = space2shape(args.observation_space[args.agent_keys[0]])
            else:
                input_dict["input_shape"] = space2shape(args.observation_space)

            if representation_name in ["Basic_MLP", "CoG_MLP"]:
                input_dict["hidden_sizes"] = args.representation_hidden_size
            else:
                if representation_name in ["Basic_CNN", "CoG_CNN", "C_DQN"]:
                    input_dict["kernels"] = args.kernels
                    input_dict["strides"] = args.strides
                    input_dict["filters"] = args.filters

            input_dict["normalize"] = None
            input_dict["initialize"] = TruncatedNormal
            input_dict["activation"] = nn.ReLU

            input_list = itemgetter(*Representation_Inputs[representation_name])(input_dict)
            if len(Representation_Inputs[representation_name]) == 1:
                return list([input_list])
            else:
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
                if policy_name in ["Categorical_AC", "Categorical_PPG", "Discrete_SAC", "Gaussian_SAC", "Gaussian_AC", "DDPG_Policy", "TD3_Policy"]:
                    input_dict["critic_hidden_size"] = args.critic_hidden_size
            input_dict["normalize"] = None
            input_dict["initialize"] = TruncatedNormal
            input_dict["activation"] = nn.ReLU
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
            except: input_dict["state_dim"] = None

            if args.env_name in ["StarCraft2", "Football"]:
                input_dict["action_space"] = args.action_space
            else:
                input_dict["action_space"] = args.action_space[args.agent_keys[0]]

            try: input_dict["n_agents"] = args.n_agents
            except: input_dict["n_agents"] = 1
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
                except: input_dict["critic_hidden_size"] = None

            input_dict["initialize"] = InitializeFunctions[args.initialize] if hasattr(args, "initialize") else None
            input_dict["normalize"] = NormalizeFunctions[args.normalize] if hasattr(args, "normalize") else None
            input_dict["activation"] = ActivationFunctions[args.activation]

            if policy_name == "Gaussian_Actor":
                input_dict["fixed_std"] = None
            input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
            return list(input_list)

