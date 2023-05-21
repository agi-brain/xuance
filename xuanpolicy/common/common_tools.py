import os
import numpy as np
import scipy.signal
import yaml
import itertools
from gym.spaces import Space, Dict
from typing import Sequence
from types import SimpleNamespace as SN
from copy import deepcopy


def recursive_dict_update(basic_dict, target_dict):
    out_dict = deepcopy(basic_dict)
    for key, value in target_dict.items():
        if isinstance(value, dict):
            out_dict[key] = recursive_dict_update(out_dict.get(key, {}), value)
        else:
            out_dict[key] = value
    return out_dict


def get_config(file_name):
    with open(file_name, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, file_name + " error: {}".format(exc)
    return config_dict


def get_arguments(agent_name, env_name, config_path=None, parser_args=None):
    """
    Get arguments from .yaml files
    agent_name: the algorithm name that will be implemented,
    env_name: env/scenario, e.g., classic/CartPole-v0,
    config_path: default is None, if None, the default configs (xuanpolicy/configs/.../*.yaml) will be loaded.
    parser_args: arguments that specified by parser tools.
    """
    main_path = os.getcwd()
    main_path_package = os.path.dirname(os.path.dirname(__file__))
    config_path_default = os.path.join(main_path_package, "configs")

    ''' get the arguments from xuanpolicy/config/basic.yaml '''
    config_basic = get_config(os.path.join(config_path_default, "basic.yaml"))

    ''' get the arguments from xuanpolicy/config/agent/env/scenario.yaml '''
    file_name = env_name + ".yaml"
    if type(agent_name) == list:
        config_algo_default = [get_config(os.path.join(config_path_default, agent, file_name)) for agent in agent_name]
        configs = [recursive_dict_update(config_basic, config_i) for config_i in config_algo_default]
        if config_path is not None:
            config_algo = get_config(os.path.join(main_path, config_path))
            configs = [recursive_dict_update(config_i, config_algo) for config_i in configs]
        if parser_args is not None:
            configs = [recursive_dict_update(config_i, parser_args.__dict__) for config_i in configs]
        args = [SN(**config_i) for config_i in configs]
    elif type(agent_name) == str:
        config_algo_default = get_config(os.path.join(config_path_default, agent_name, file_name))
        configs = recursive_dict_update(config_basic, config_algo_default)
        if config_path is not None:
            config_algo = get_config(os.path.join(main_path, config_path))
            configs = recursive_dict_update(configs, config_algo)
        if parser_args is not None:
            configs = recursive_dict_update(configs, parser_args.__dict__)
        args = SN(**configs)
    else:
        raise "Unsupported agent_name or env_name!"
    return args


def get_runner(agent_name,
               env_name,
               config_path=None,
               parser_args=None,
               is_test=False):
    """
    This method returns a runner that specified by the users according to the inputs:
    agent_name: the algorithm name that will be implemented,
    env_name: env/scenario, e.g., classic/CartPole-v0,
    config_path: default is None, if None, the default configs (xuanpolicy/configs/.../*.yaml) will be loaded.
    parser_args: arguments that specified by parser tools.
    is_test: default is False, if True, it will load the models and run the environment with rendering.
    """
    args = get_arguments(agent_name, env_name, config_path, parser_args)

    device = args[0].device if type(args) == list else args.device
    dl_toolbox = args[0].dl_toolbox if type(args) == list else args.dl_toolbox
    print("Calculating device:", device)

    if dl_toolbox == "torch":
        from xuanpolicy.xuanpolicy_torch.runners import REGISTRY as run_REGISTRY
        print("Deep learning toolbox: PyTorch.")
    elif dl_toolbox == "mindspore":
        from xuanpolicy.xuanpolicy_ms.runners import REGISTRY as run_REGISTRY
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
        from xuanpolicy.xuanpolicy_tf.runners import REGISTRY as run_REGISTRY
        print("Deep learning toolbox: TensorFlow.")
        if device in ["cpu", "CPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if dl_toolbox == '':
            raise AttributeError("You have to assign a deep learning toolbox")
        else:
            raise AttributeError("Cannot find a deep learning toolbox named " + dl_toolbox)

    if type(args) == list:
        for i_alg in range(len(agent_name)):
            args[i_alg].agent_name = agent_name[i_alg]
            notation = args[i_alg].dl_toolbox + '/'
            args[i_alg].modeldir = os.path.join(os.getcwd(), args[i_alg].modeldir + notation + args[i_alg].env_id + '/')
            args[i_alg].logdir = args[i_alg].logdir + notation + args[i_alg].env_id + '/'
            if is_test is not None:
                args[i_alg].test_mode = int(is_test)
                args[i_alg].parallels = 1
        print("Algorithm:", *[arg.agent for arg in args])
        print("Environment:", args[0].env_name)
        print("Scenario:", args[0].env_id)
    else:
        args.agent_name = agent_name
        notation = args.dl_toolbox + '/'
        args.modeldir = os.path.join(os.getcwd(), args.modeldir + notation + args.env_id + '/')
        args.logdir = args.logdir + notation + args.env_id + '/'
        if is_test is not None:
            args.test_mode = int(is_test)
            args.parallels = 1
        print("Algorithm:", args.agent)
        print("Environment:", args.env_name)
        print("Scenario:", args.env_id)

    runner = run_REGISTRY[args[0].runner](args) if type(args) == list else run_REGISTRY[args.runner](args)
    return runner


def create_directory(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def space2shape(observation_space: Space):
    if isinstance(observation_space, Dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    else:
        return observation_space.shape


def dict_reshape(keys, dict_list: Sequence[dict]):
    results = {}
    for key in keys():
        results[key] = np.array([element[key] for element in dict_list], np.float32)
    return results


def discount_cumsum(x, discount=0.99):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def merge_iterators(self, *iters):
    itertools.chain(*iters)
