Common Tools
==============================================

The common tools serve to prepare the DRL model before training,
such as loading hyper parameters from YAML files, obtaining terminal commands,
and creating a predefined runner for DRL implementations.


.. py:function::
    xuance.common.common_tools.recursive_dict_update(basic_dict, target_dict)

    Update the dict values.

    :param basic_dict: the original dict variable that to be updated.
    :type basic_dict: Dict
    :param target_dict: the target dict variable with new values.
    :type target_dict: Dict
    :return: A dict mapping keys of basic_dict to the values of the same keys in target_dict.
    :rtype: Dict

.. py:function::
    xuance.common.common_tools.get_config(file_dir)

    Get dict variable from a YAML file.

    :param file_dir: the directory of the YAML file.
    :type file_name: str
    :return: the keys and corresponding values in the YAML file.
    :rtype: Dict

.. py:function::
    xuance.common.common_tools.get_arguments(method, env, env_id, config_path=None, parser_args=None)

    Get arguments from .yaml files.

    :param method: the algorithm name that will be implemented.
    :type method: str
    :param env: The name of the environment.
    :type env: str
    :param env_id: The name of the scenario in the environment.
    :type env_id: str
    :param config_path: default is None, if None, the default configs (xuance/configs/.../\*.yaml) will be loaded.
    :type config_path: str
    :param parser_args: arguments that specified by parser tools.
    :type parser_args: Dict
    :return: the SimpleNamespace variables that contains attributes for DRL implementations.
    :rtype: SimpleNamespace

.. py:function::
    xuance.common.common_tools.get_runner(method, env, env_id, config_path=None, parser_args=None, is_test=None)

    This method returns a runner that specified by the users according to the inputs.

    :param method: the algorithm name that will be implemented.
    :type method: str
    :param env: The name of the environment.
    :type env: str
    :param env_id: The name of the scenario in the environment.
    :type env_id: str
    :param config_path: Default is None, if None, the default configs (xuance/configs/.../\*.yaml) will be loaded.
    :type config_path: str
    :param parser_args: Arguments that specified by parser tools.
    :type parser_args: Dict
    :param is_test: Default is False, if True, it will load the models and run the environment with rendering.
    :type is_test: bool
    :return: An implementation of a runner that enables to run the DRL algorithms.
    :rtype: object

.. py:function::
    xuance.common.common_tools.combined_shape(length, shape)

    Expand the original shape.

    :param length: The length of first dimension to expand.
    :type length: int
    :param shape: The target shape to be expanded.
    :type shape: None, tuple, list, int
    :return: A new shape that is expanded from shape.
    :rtype: tuple

.. py:function::
    xuance.common.common_tools.space2shape(observation_space)

    Convert gym.space variable to shape.

    :param observation_space: the space variable with type of gym.Space.
    :type observation_space: Space
    :return: The shape of the observation_space.
    :rtype: tuple

.. py:function::
    xuance.common.common_tools.discount_cumsum(x, discount)

    Get a discounted cumulated summation.

    :param x: The original sequence. In DRL, x can be reward sequence.
    :type x: np.ndarray, list
    :param discount: the discount factor (gamma), default is 0.99.
    :type discount: float
    :return: The discounted cumulative returns for each step.
    :rtype: np.ndarray, list


.. raw:: html

    <br><hr>


Source Code
-----------------

.. code-block:: python

    import os
    import yaml
    import numpy as np
    import scipy.signal
    from copy import deepcopy
    from gym.spaces import Space, Dict
    from types import SimpleNamespace as SN
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


    def get_config(file_dir):
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


    def get_arguments(method, env, env_id, config_path=None, parser_args=None):
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
        config_basic = get_config(os.path.join(config_path_default, "basic.yaml"))

        ''' get the arguments from, e.g., xuance/config/dqn/box2d/CarRacing-v2.yaml '''
        if type(method) == list:  # for different groups of MARL algorithms.
            file_name = env + "/" + env_id + ".yaml"
            config_algo_default = [get_config(os.path.join(config_path_default, agent, file_name)) for agent in method]
            configs = [recursive_dict_update(config_basic, config_i) for config_i in config_algo_default]
            if config_path is not None:
                config_algo = [get_config(os.path.join(main_path, _path)) for _path in config_path]
                configs = [recursive_dict_update(config_i, config_algo[i]) for i, config_i in enumerate(configs)]
            if parser_args is not None:
                configs = [recursive_dict_update(config_i, parser_args.__dict__) for config_i in configs]
            args = [SN(**config_i) for config_i in configs]
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
                    raise RuntimeError(
                        f"The file of '{error_path_env_id}' or '{error_path_env}' does not exist in this library. "
                        f"You can also customize the configuration file by specifying the `config_path` parameter "
                        f"in the `get_runner()` function.")
            else:
                config_path = os.path.join(main_path, config_path)
            config_algo_default = get_config(config_path)
            configs = recursive_dict_update(config_basic, config_algo_default)
            # load parser_args and rewrite the parameters if their names are same.
            if parser_args is not None:
                configs = recursive_dict_update(configs, parser_args.__dict__)
            if not ('env_id' in configs.keys()):
                configs['env_id'] = env_id
            args = SN(**configs)
        else:
            raise RuntimeError("Unsupported agent_name or env_name!")
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
            env: env/scenario, e.g., classic/CartPole-v0,
            config_path: default is None, if None, the default configs (xuance/configs/.../*.yaml) will be loaded.
            parser_args: arguments that specified by parser tools.
            is_test: default is False, if True, it will load the models and run the environment with rendering.

        Returns:
            An implementation of a runner that enables to run the DRL algorithms.
        """
        args = get_arguments(method, env, env_id, config_path, parser_args)

        device = args[0].device if type(args) == list else args.device
        dl_toolbox = args[0].dl_toolbox if type(args) == list else args.dl_toolbox
        print("Calculating device:", device)

        if dl_toolbox == "torch":
            from xuance.torch.runners import REGISTRY as run_REGISTRY
            print("Deep learning toolbox: PyTorch.")
        elif dl_toolbox == "mindspore":
            from xuance.mindspore.runners import REGISTRY as run_REGISTRY
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
            from xuance.tensorflow.runners import REGISTRY as run_REGISTRY
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

                if ('model_dir' in args.__dict__) and ('log_dir' in args[i_alg].__dict__):
                    args[i_alg].model_dir = os.path.join(os.getcwd(),
                                                        args[i_alg].model_dir + notation + args[i_alg].env_id + '/')
                    args[i_alg].log_dir = args[i_alg].log_dir + notation + args[i_alg].env_id + '/'
                else:
                    if config_path is not None:
                        raise RuntimeError(f"'model_dir' or 'log_dir' is not defined in {config_path} files.")
                    elif method[i_alg] not in method_list.keys():
                        raise RuntimeError(f"The method named '{method[i_alg]}' is currently not supported in XuanCe.")
                    elif args[i_alg].env not in method_list[method[i_alg]]:
                        raise RuntimeError(
                            f"The environment named '{args[i_alg].env}' is currently not supported for {method_list[method[i_alg]]}.")
                    else:
                        print("Failed to load arguments for the implementation!")

                if is_test:
                    args[i_alg].test_mode = int(is_test)
                    args[i_alg].parallels = 1

            # print("Algorithm:", *[arg.agent for arg in args])
            print("Algorithm:", *agents_name_string)
            print("Environment:", args[0].env_name)
            print("Scenario:", args[0].env_id)
            for arg in args:
                if arg.agent_name != "random":
                    runner = run_REGISTRY[arg.runner](args)
                    return runner
            raise "Both sides of policies are random!"
        else:
            args.agent_name = method
            notation = args.dl_toolbox + '/'
            if ('model_dir' in args.__dict__) and ('log_dir' in args.__dict__):
                args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.dl_toolbox, args.env_id)
                args.log_dir = os.path.join(args.log_dir, notation, args.env_id)
            else:
                if config_path is not None:
                    raise RuntimeError(f"'model_dir' or 'log_dir' is not defined in {config_path} file.")
                elif args.method not in method_list.keys():
                    raise RuntimeError(f"The method named '{args.method}' is currently not supported in XuanCe.")
                elif args.env not in method_list[args.method]:
                    raise RuntimeError(f"The environment named '{args.env}' is currently not supported for {args.method}.")
                else:
                    print("Failed to load arguments for the implementation!")

            if is_test:
                args.test_mode = int(is_test)
                args.parallels = 1
            print("Algorithm:", args.agent)
            print("Environment:", args.env_name)
            print("Scenario:", args.env_id)
            runner = run_REGISTRY[args[0].runner](args) if type(args) == list else run_REGISTRY[args.runner](args)
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


    def space2shape(observation_space: Space):
        """Convert gym.space variable to shape
        Args:
            observation_space: the space variable with type of gym.Space

        Returns:
            The shape of the observation_space.
        """
        if isinstance(observation_space, Dict):
            return {key: observation_space[key].shape for key in observation_space.keys()}
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

