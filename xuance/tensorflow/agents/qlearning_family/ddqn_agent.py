from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.tensorflow.agents.qlearning_family.dqn_agent import DQN_Agent


class DDQN_Agent(DQN_Agent):
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(DDQN_Agent, self).__init__(config, envs, observation_space, action_space, callback)
