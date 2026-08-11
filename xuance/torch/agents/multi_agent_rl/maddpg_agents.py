from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents.multi_agent_rl import IDDPG_Agents
from xuance.torch.rl_models import DeterministicActor, CentralizedActionValueCritic
from xuance.torch.rl_models.architectures import MultiAgentDeterministicActorCritic


class MADDPG_Agents(IDDPG_Agents):
    """The implementation of MADDPG agents.

    Args:
        config: The Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        super(MADDPG_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        """
                Build the MARL model.

                Returns:
                    model (torch.nn.Module): The MARL model.
                """
        actor_networks = ModuleDict()
        critic_networks = ModuleDict()
        joint_obs_space = (sum([sum(self.observation_space[k].shape) for k in self.agent_keys]),)
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as actor representations
            actor_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared actor-network
            actor_networks[group_key] = DeterministicActor(
                representation=actor_feature_encoder,
                actor_hidden_size=self.config.actor_hidden_size,
                action_space=self.action_space[reference_agent],
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=self.device
            )
            # build critic feature encoder as critic representations
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=joint_obs_space
            )
            # build inner-group shared critic-network
            critic_networks[group_key] = CentralizedActionValueCritic(
                representation=critic_feature_encoder,
                action_space=self.action_space,
                critic_hidden_size=self.config.critic_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build the RL model
        model = MultiAgentDeterministicActorCritic(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model
