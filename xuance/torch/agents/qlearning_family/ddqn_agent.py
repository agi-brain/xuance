from argparse import Namespace
from xuance.common import Union, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents import BaseCallback
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    """The implementation of Double DQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(DDQN_Agent, self).__init__(config, envs, callback)

