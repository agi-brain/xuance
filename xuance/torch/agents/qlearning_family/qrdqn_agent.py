from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from xuance.torch.rl_models.architectures import QRDeepQNetwork


class QRDQN_Agent(DQN_Agent):
    """The implementation of QRDQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(QRDQN_Agent, self).__init__(config, envs, observation_space, action_space, callback)

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build RL model.
        model = QRDeepQNetwork(
            representation=representation,
            hidden_size=self.config.q_hidden_size,
            action_space=self.action_space,
            quantile_num=self.config.quantile_num,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model
