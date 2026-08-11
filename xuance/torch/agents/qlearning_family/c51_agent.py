from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from xuance.torch.rl_models.architectures import C51DeepQNetwork


class C51_Agent(DQN_Agent):
    """The implementation of C51DQN agent.

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
        super(C51_Agent, self).__init__(config, envs, observation_space, action_space, callback)

    def _build_model(self) -> Module:
        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        model = C51DeepQNetwork(
            representation=representation,
            hidden_size=self.config.q_hidden_size,
            action_space=self.action_space,
            atom_num=self.config.atom_num,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model

