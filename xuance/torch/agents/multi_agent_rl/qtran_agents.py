from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import List, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.agents import OffPolicyMARLAgents
from xuance.torch.rl_models import DiscreteActionValueCritic
from xuance.torch.rl_models.heads import VDN_Mixer, QTRAN_Base, QTRAN_Alt
from xuance.torch.rl_models.architectures import QTranMixingNetwork


class QTRAN_Agents(OffPolicyMARLAgents):
    """The implementation of QTRAN agents.

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
        super(QTRAN_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.state_space = envs.state_space
        self.use_global_state = True

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        # build policy, optimizers, schedulers
        self.model = self._build_model()  # build the MARL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.agent_grouping, self.model, self.callback)

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        q_networks = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as representations
            agent_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build inner-group shared q-network
            q_networks[group_key] = DiscreteActionValueCritic(
                representation=agent_feature_encoder,
                action_space=self.action_space[reference_agent],
                critic_hidden_size=self.config.q_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build mixers
        mixer = VDN_Mixer()

        input_qtran_mixer = dict(
            dim_state=self.state_space.shape[-1],
            action_space=self.action_space,
            dim_hidden=self.config.qtran_net_hidden_dim,
            n_agents=self.config.n_agents,
            dim_utility_hidden=self.config.q_hidden_size[0],
            use_parameter_sharing=self.use_parameter_sharing,
            device=self.device
        )
        if self.config.agent == "QTRAN_base":
            qtran_mixer = QTRAN_Base(**input_qtran_mixer)
        elif self.config.agent == "QTRAN_alt":
            qtran_mixer = QTRAN_Alt(**input_qtran_mixer)
        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        model = QTranMixingNetwork(
            grouping=self.agent_grouping,
            q_networks=q_networks,
            mixer=mixer,
            qtran_mixer=qtran_mixer,
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training
        )

        return model
