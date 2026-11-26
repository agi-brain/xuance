import gymnasium
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents.multi_agent_rl.isac_agents import ISAC_Agents
from xuance.torch.rl_models import (CategoricalActor, SAC_GaussianActor,
                                    TwinCentralizedActionValueCritic, TwinDiscreteCentralizedActionValueCritic)
from xuance.torch.rl_models.architectures import MultiAgentSoftActorCritic, MultiAgentSoftActorCriticDiscrete


class MASAC_Agents(ISAC_Agents):
    """The implementation of MASAC agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
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
        super(MASAC_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        actor_input = dict(
            actor_hidden_size=self.config.actor_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )
        critic_input = dict(
            critic_hidden_size=self.config.critic_hidden_size,
            normalizer=self.normalize_fn,
            initializer=self.initializer,
            activation=self.activation,
            device=self.device
        )
        if isinstance(self.action_space[self.agent_keys[0]], gymnasium.spaces.Box):
            Actor = SAC_GaussianActor
            Critic = TwinCentralizedActionValueCritic
            actor_input['activation_action'] = ActivationFunctions[self.config.activation_action]
            self.continuous_control = True
            Architecture = MultiAgentSoftActorCritic
        elif isinstance(self.action_space[self.agent_keys[0]], gymnasium.spaces.Discrete):
            Actor = CategoricalActor
            Critic = TwinDiscreteCentralizedActionValueCritic
            self.continuous_control = False
            Architecture = MultiAgentSoftActorCriticDiscrete
        else:
            raise NotImplementedError

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
            actor_input['representation'] = actor_feature_encoder
            actor_input['action_space'] = self.action_space[reference_agent]
            actor_networks[group_key] = Actor(**actor_input)
            # build critic feature encoder as critic representations
            critic_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=joint_obs_space
            )
            critic_input['representation'] = critic_feature_encoder
            critic_input['action_space'] = self.action_space
            if not self.continuous_control:
                critic_input['action_space_key'] = reference_agent
            # build inner-group shared critic-network
            critic_networks[group_key] = Critic(**critic_input)

        # build the RL model
        model = Architecture(
            grouping=self.agent_grouping,
            actors=actor_networks,
            critics=critic_networks,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model
