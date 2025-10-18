from argparse import Namespace
from xuance.environment.utils import XuanCeEnvWrapper, XuanCeAtariEnvWrapper, XuanCeMultiAgentEnvWrapper
from xuance.environment.utils import RawEnvironment, RawMultiAgentEnv
from xuance.environment.vector_envs import DummyVecEnv, DummyVecEnv_Atari, DummyVecMultiAgentEnv
from xuance.environment.vector_envs import SubprocVecEnv, SubprocVecEnv_Atari, SubprocVecMultiAgentEnv
from xuance.environment.single_agent_env import REGISTRY_ENV
from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
from xuance.environment.vector_envs import REGISTRY_VEC_ENV


def make_envs(config: Namespace):
    """
    Creates and returns a set of environments based on the provided configuration.

    This function supports single-agent, multi-agent, and vectorized environments and handles
    the initialization of the environment(s) based on the configuration settings. The function
    also manages distributed training setups and environment vectorization.

    Parameters:
    -----------
    config : Namespace
        A configuration object containing the necessary settings to initialize the environment.
        The configuration should contain the following attributes:
        - env_name (str): The name of the environment to create.
        - env_seed (int): The seed value for environment initialization.
        - distributed_training (bool): Whether to use distributed training.
        - parallels (int): The number of parallel environments for vectorized setups.
        - vectorize (str): The type of vectorization to apply (e.g., 'DummyVecEnv', 'SubprocVecEnv', etc.).

    Returns:
        List of environments based on the configuration settings.
    """
    def _thunk(env_seed: int = None):
        """
        Function that creates and returns an environment based on the config settings.

        Parameters:
        -----------
        env_seed : int, optional
            The seed to use for environment initialization. Defaults to `None`.

        Returns:
        --------
        environment
            The created environment based on the configuration settings (single-agent or multi-agent).
        """
        config.env_seed = env_seed
        if config.env_name in REGISTRY_ENV.keys():
            if config.env_name == "Platform":
                return REGISTRY_ENV[config.env_name](config)
            elif config.env_name == "Atari":
                return XuanCeAtariEnvWrapper(REGISTRY_ENV[config.env_name](config))
            else:
                return XuanCeEnvWrapper(REGISTRY_ENV[config.env_name](config))
        elif config.env_name in REGISTRY_MULTI_AGENT_ENV.keys():
            return XuanCeMultiAgentEnvWrapper(REGISTRY_MULTI_AGENT_ENV[config.env_name](config))
        else:
            raise AttributeError(f"The environment named {config.env_name} cannot be created.")

    distributed_training = config.distributed_training if hasattr(config, "distributed_training") else False
    if not hasattr(config, "render_mode"):
        config.render_mode = "human"

    if distributed_training:
        # rank = int(os.environ['RANK'])  # for torch.nn.parallel.DistributedDataParallel
        rank = 1
        config.env_seed += rank * config.parallels

    if config.vectorize in REGISTRY_VEC_ENV.keys():
        env_fn = [_thunk for _ in range(config.parallels)]
        return REGISTRY_VEC_ENV[config.vectorize](env_fn, config.env_seed)
    elif config.vectorize == "NOREQUIRED":
        return _thunk()
    else:
        raise AttributeError(f"The vectorizer {config.vectorize} is not implemented.")
