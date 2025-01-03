from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.mindspore.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(DDQN_Agent, self).__init__(config, envs)
