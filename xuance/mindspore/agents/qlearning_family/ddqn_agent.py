from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.mindspore.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DDQN_Agent, self).__init__(config, envs)
