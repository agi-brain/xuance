from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    """The implementation of Double DQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DDQN_Agent, self).__init__(config, envs)

