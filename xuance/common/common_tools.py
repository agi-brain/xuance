import os
import yaml
import time
import numpy as np
import scipy.signal
from copy import deepcopy
from types import SimpleNamespace as SN
from typing import Dict
from xuance.configs import method_list

EPS = 1e-8


def recursive_dict_update(basic_dict, target_dict):
    """Update the dict values.
    Args:
        basic_dict: the original dict variable that to be updated.
        target_dict: the target dict variable with new values.

    Returns:
        A dict mapping keys of basic_dict to the values of the same keys in target_dict.
        For example:

        basic_dict = {'a': 1, 'b': 2}
        target_dict = {'a': 3, 'c': 4}
        out_dict = recursive_dict_update(basic_dict, target_dict)

        output_dict = {'a': 3, 'b': 2, 'c': 4}
    """
    out_dict = deepcopy(basic_dict)
    for key, value in target_dict.items():
        if isinstance(value, dict):
            out_dict[key] = recursive_dict_update(out_dict.get(key, {}), value)
        else:
            out_dict[key] = value
    return out_dict


def get_configs(file_dir):
    """Get dict variable from a YAML file.
    Args:
        file_dir: the directory of the YAML file.

    Returns:
        config_dict: the keys and corresponding values in the YAML file.
    """
    with open(file_dir, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, file_dir + " error: {}".format(exc)
    return config_dict


def get_arguments(method, env, env_id, config_path=None, parser_args=None, is_test=False):
    """Get arguments from .yaml files
    Args:
        method: the algorithm name that will be implemented,
        env: The name of the environment,
        env_id: The name of the scenario in the environment.
        config_path: default is None, if None, the default configs (xuance/configs/.../*.yaml) will be loaded.
        parser_args: arguments that specified by parser tools.

    Returns:
        args: the SimpleNamespace variables that contains attributes for DRL implementations.
    """
    main_path = os.getcwd()
    main_path_package = os.path.dirname(os.path.dirname(__file__))
    config_path_default = os.path.join(main_path_package, "configs")

    ''' get the arguments from xuance/config/basic.yaml '''
    config_basic = get_configs(os.path.join(config_path_default, "basic.yaml"))

    ''' get the arguments from, e.g., xuance/config/dqn/box2d/CarRacing-v2.yaml '''
    if type(method) == list:  # for different groups of MARL algorithms.
        if config_path is None:
            config_path = []
            file_name_env_id = env + "/" + env_id + ".yaml"
            file_name_env = env + "/" + env_id + ".yaml"
            config_path_env_id = [os.path.join(config_path_default, agent, file_name_env_id) for agent in method]
            config_path_env = [os.path.join(config_path_default, agent, file_name_env) for agent in method]
            for i_agent, agent in enumerate(method):
                if os.path.exists(config_path_env_id[i_agent]):
                    config_path.append(config_path_env_id[i_agent])
                elif os.path.exists(config_path_env[i_agent]):
                    config_path.append(config_path_env[i_agent])
                else:
                    raise AttributeError(
                        f"Cannot find file named '{config_path_env_id[i_agent]}' or '{config_path_env[i_agent]}'."
                        f"You can also customize the configuration file by specifying the `config_path` parameter "
                        f"in the `get_runner()` function.")
        else:
            config_path = [os.path.join(main_path, _path) for _path in config_path]
        config_algo_default = [get_configs(_path) for _path in config_path]
        configs = [recursive_dict_update(config_basic, config_i) for config_i in config_algo_default]
        # load parser_args and rewrite the parameters if their names are same.
        if parser_args is not None:
            configs = [recursive_dict_update(config_i, parser_args.__dict__) for config_i in configs]
        args = [SN(**config_i) for config_i in configs]
        if is_test:  # for test mode
            for i_args in range(len(args)):
                args[i_args].test_mode = int(is_test)
                args[i_args].parallels = 1
    elif type(method) == str:
        if config_path is None:
            file_name_env_id = env + "/" + env_id + ".yaml"
            file_name_env = env + ".yaml"
            config_path_env_id = os.path.join(config_path_default, method, file_name_env_id)
            config_path_env = os.path.join(config_path_default, method, file_name_env)
            if os.path.exists(config_path_env_id):
                config_path = config_path_env_id
            elif os.path.exists(config_path_env):
                config_path = config_path_env
            else:
                error_path_env_id = os.path.join('./xuance/configs', method, file_name_env_id)
                error_path_env = os.path.join('./xuance/configs', method, file_name_env)
                raise AttributeError(
                    f"The file '{error_path_env_id}' or '{error_path_env}' does not exist in this library. "
                    f"You can also customize the configuration file by specifying the `config_path` parameter "
                    f"in the `get_runner()` function.")
        else:
            config_path = os.path.join(main_path, config_path)
        config_algo_default = get_configs(config_path)
        configs = recursive_dict_update(config_basic, config_algo_default)
        # load parser_args and rewrite the parameters if their names are same.
        if parser_args is not None:
            configs = recursive_dict_update(configs, parser_args.__dict__)
        if not ('env_id' in configs.keys()):
            configs['env_id'] = env_id
        args = SN(**configs)
        if is_test:
            args.test_mode = int(is_test)
            args.parallels = 1
    else:
        raise AttributeError("Unsupported agent_name or env_name!")
    return args


def get_runner(method,
               env,
               env_id,
               config_path=None,
               parser_args=None,
               is_test=False):
    """
    This method returns a runner that specified by the users according to the inputs.
    Args:
        method: the algorithm name that will be implemented,
        env: The name of the environment,
        env_id: The name of the scenario in the environment.
        config_path: default is None, if None, the default configs (xuance/configs/.../*.yaml) will be loaded.
        parser_args: arguments that specified by parser tools.
        is_test: default is False, if True, it will load the models and run the environment with rendering.

    Returns:
        An implementation of a runner that enables to run the DRL algorithms.
    """
    args = get_arguments(method, env, env_id, config_path, parser_args, is_test)

    device = args[0].device if type(args) == list else args.device
    dl_toolbox = args[0].dl_toolbox if type(args) == list else args.dl_toolbox
    print("Calculating device:", device)

    if dl_toolbox == "torch":
        from xuance.torch.runners import REGISTRY_Runner
        print("Deep learning toolbox: PyTorch.")
    elif dl_toolbox == "mindspore":
        from xuance.mindspore.runners import REGISTRY_Runner
        from mindspore import context
        print("Deep learning toolbox: MindSpore.")
        if device != "Auto":
            if device in ["cpu", "CPU", "gpu", "GPU"]:
                device = "CPU"
            context.set_context(device_target=device)
        # context.set_context(enable_graph_kernel=True)
        context.set_context(mode=context.GRAPH_MODE)  # 静态图（断点无法进入）
        # context.set_context(mode=context.PYNATIVE_MODE)  # 动态图（便于调试）
    elif dl_toolbox == "tensorflow":
        from xuance.tensorflow.runners import REGISTRY_Runner
        print("Deep learning toolbox: TensorFlow.")
        if device in ["cpu", "CPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if dl_toolbox == '':
            raise AttributeError("You have to assign a deep learning toolbox")
        else:
            raise AttributeError("Cannot find a deep learning toolbox named " + dl_toolbox)

    if type(args) == list:
        agents_name_string = []
        for i_alg in range(len(method)):
            if i_alg < len(method) - 1:
                agents_name_string.append(args[i_alg].agent + " vs")
            else:
                agents_name_string.append(args[i_alg].agent)
            args[i_alg].agent_name = method[i_alg]
            notation = args[i_alg].dl_toolbox + '/'

            if ('model_dir' in args[i_alg].__dict__) and ('log_dir' in args[i_alg].__dict__):
                args[i_alg].model_dir = os.path.join(os.getcwd(),
                                                     args[i_alg].model_dir + notation + args[i_alg].env_id + '/')
                args[i_alg].log_dir = args[i_alg].log_dir + notation + args[i_alg].env_id + '/'
            else:
                if config_path is not None:
                    raise AttributeError(f"'model_dir' or 'log_dir' is not defined in {config_path} files.")
                elif method[i_alg] not in method_list.keys():
                    raise AttributeError(f"The method named '{method[i_alg]}' is currently not supported in XuanCe.")
                elif args[i_alg].env not in method_list[method[i_alg]]:
                    raise AttributeError(
                        f"The environment named '{args[i_alg].env}' is currently not supported for {method_list[method[i_alg]]}.")
                else:
                    print("Failed to load arguments for the implementation!")

        # print("Algorithm:", *[arg.agent for arg in args])
        print("Algorithm:", *agents_name_string)
        print("Environment:", args[0].env_name)
        print("Scenario:", args[0].env_id)
        for arg in args:
            if arg.agent_name != "random":
                runner = REGISTRY_Runner[arg.runner](args)
                return runner
        raise AttributeError("Both sides of policies are random!")
    else:
        args.agent_name = method
        notation = args.dl_toolbox + '/'
        if ('model_dir' in args.__dict__) and ('log_dir' in args.__dict__):
            args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.dl_toolbox, args.env_id)
            args.log_dir = os.path.join(args.log_dir, notation, args.env_id)
        else:
            if config_path is not None:
                raise AttributeError(f"'model_dir' or 'log_dir' is not defined in {config_path} file.")
            elif args.method not in method_list.keys():
                raise AttributeError(f"The method named '{args.method}' is currently not supported in XuanCe.")
            elif args.env not in method_list[args.method]:
                raise AttributeError(f"The environment named '{args.env}' is currently not supported for {args.method}.")
            else:
                print("Failed to load arguments for the implementation!")

        print("Algorithm:", args.agent)
        print("Environment:", args.env_name)
        print("Scenario:", args.env_id)
        runner = REGISTRY_Runner[args.runner](args)
        return runner


def create_directory(path):
    """Create an empty directory.
    Args:
        path: the path of the directory
    """
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


def combined_shape(length, shape=None):
    """Expand the original shape.
    Args:
        length: the length of first dimension to expand.
        shape: the target shape to be expanded.

    Returns:
        A new shape that is expanded from shape.

    Examples
    --------
        >>> length = 2
        >>> shape_1 = None
        >>> shape_2 = 3
        >>> shape_3 = [4, 5]
        >>> combined(length, shape_1)
            (2, )
        >>> combined(length, shape_2)
            (2, 3)
        >>> combined(length, shape_3)
            (2, 4, 5)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def space2shape(observation_space):
    """Convert gym.space variable to shape
    Args:
        observation_space: the space variable with type of gym.Space.

    Returns:
        The shape of the observation_space.
    """
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape


def discount_cumsum(x, discount=0.99):
    """Get a discounted cumulated summation.
    Args:
        x: The original sequence. In DRL, x can be reward sequence.
        discount: the discount factor (gamma), default is 0.99.

    Returns:
        The discounted cumulative returns for each step.

    Examples
    --------
    >>> x = [0, 1, 2, 2]
    >>> y = discount_cumsum(x, discount=0.99)
    [4.890798, 4.9402, 3.98, 2.0]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_time_string():
    t_now = time.localtime(time.time())
    t_year = str(t_now.tm_year).zfill(4)
    t_month = str(t_now.tm_mon).zfill(2)
    t_day = str(t_now.tm_mday).zfill(2)
    t_hour = str(t_now.tm_hour).zfill(2)
    t_min = str(t_now.tm_min).zfill(2)
    t_sec = str(t_now.tm_sec).zfill(2)
    time_string = f"{t_year}_{t_month}{t_day}_{t_hour}{t_min}{t_sec}"
    return time_string

