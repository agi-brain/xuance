import os
import yaml
import time
import scipy.signal
from copy import deepcopy
from types import SimpleNamespace as SN

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


def load_yaml(file_dir) -> dict:
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


def get_arguments(algo, env, env_id, config_path=None, parser_args=None):
    """Get arguments from ``.yaml`` files
    Args:
        algo: the algorithm name that will be implemented,
        env: The name of the environment,
        env_id: The name of the scenario in the environment.
        config_path: default is None, if None, the default configs (``xuance/configs/.../*.yaml``) will be loaded.
        parser_args: arguments that specified by parser tools.

    Returns:
        args: the SimpleNamespace variables that contains attributes for DRL implementations.
    """
    main_path = os.getcwd()
    main_path_package = os.path.dirname(os.path.dirname(__file__))
    config_path_default = os.path.join(main_path_package, "configs")

    ''' get the arguments from xuance/config/basic.yaml '''
    config_basic = load_yaml(os.path.join(config_path_default, "basic.yaml"))

    ''' get the arguments from, e.g., xuance/config/dqn/box2d/CarRacing-v3.yaml '''
    if type(algo) == list:  # for different groups of MARL algorithms.
        if config_path is None:
            config_path = []
            file_name_env_id = env + "/" + env_id + ".yaml"
            file_name_env = env + "/" + env_id + ".yaml"
            config_path_env_id = [os.path.join(config_path_default, agent, file_name_env_id) for agent in algo]
            config_path_env = [os.path.join(config_path_default, agent, file_name_env) for agent in algo]
            for i_agent, agent in enumerate(algo):
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
        config_algo_default = [load_yaml(_path) for _path in config_path]
        configs = [recursive_dict_update(config_basic, config_i) for config_i in config_algo_default]
        # load parser_args and rewrite the parameters if their names are same.
        if parser_args is not None:
            configs = [recursive_dict_update(config_i, parser_args.__dict__) for config_i in configs]
        args = [SN(**config_i) for config_i in configs]
        for arg in args:
            if arg.dl_toolbox == "torch":
                from xuance.torch import set_device
            elif arg.dl_toolbox == "tensorflow":
                from xuance.tensorflow import set_device
            elif arg.dl_toolbox == "mindspore":
                from xuance.mindspore import set_device
            else:
                raise f"Unsupported dl_toolbox: {arg}"
            arg.device = set_device(arg.device)
    elif type(algo) == str:
        if config_path is None:
            file_name_env_id = env + "/" + env_id + ".yaml"
            file_name_env = env + ".yaml"
            config_path_env_id = os.path.join(config_path_default, algo, file_name_env_id)
            config_path_env = os.path.join(config_path_default, algo, file_name_env)
            if os.path.exists(config_path_env_id):
                config_path = config_path_env_id
            elif os.path.exists(config_path_env):
                config_path = config_path_env
            else:
                error_path_env_id = os.path.join('./xuance/configs', algo, file_name_env_id)
                error_path_env = os.path.join('./xuance/configs', algo, file_name_env)
                raise AttributeError(
                    f"The file '{error_path_env_id}' or '{error_path_env}' does not exist in this library. "
                    f"You can also customize the configuration file by specifying the `config_path` parameter "
                    f"in the `get_runner()` function.")
        else:
            config_path = os.path.join(main_path, config_path)
        config_algo_default = load_yaml(config_path)
        configs = recursive_dict_update(config_basic, config_algo_default)
        # load parser_args and rewrite the parameters if their names are same.
        if parser_args is not None:
            configs = recursive_dict_update(configs, parser_args.__dict__)
        if not ('env_id' in configs.keys()):
            configs['env_id'] = env_id
        args = SN(**configs)
        if args.dl_toolbox == "torch":
            from xuance.torch import set_device
        elif args.dl_toolbox == "tensorflow":
            from xuance.tensorflow import set_device
        elif args.dl_toolbox == "mindspore":
            from xuance.mindspore import set_device
        else:
            raise f"Unsupported dl_toolbox: {args}"
        args.device = set_device(args.device)
    else:
        raise AttributeError("Unsupported agent_name or env_name!")
    return args


def create_directory(path):
    """Create an empty directory.
    Args:
        path: the path of the directory
    """
    """Create a directory (recursively) if it does not exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def discount_cumsum(x, discount=0.99):
    """Get a discounted cumulated summation.
    Args:
        x: The original sequence. In DRL, x can be reward sequence.
        discount: the discount factor (gamma), default is 0.99.

    Returns:
        The discounted cumulative returns for each step.

    Examples:
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
