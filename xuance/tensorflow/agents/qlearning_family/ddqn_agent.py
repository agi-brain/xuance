from argparse import Namespace
from xuance.common import Union, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow.agents import BaseCallback
from xuance.tensorflow.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv],
                 callback: Optional[BaseCallback] = None):
        super(DDQN_Agent, self).__init__(config, envs, callback)
