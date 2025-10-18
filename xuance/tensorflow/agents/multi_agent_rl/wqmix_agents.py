from argparse import Namespace
from xuance.common import Union, Optional, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.tensorflow import Module
from xuance.tensorflow.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.tensorflow.policies import REGISTRY_Policy, QMIX_mixer, QMIX_FF_mixer
from xuance.tensorflow.agents.multi_agent_rl.qmix_agents import QMIX_Agents


class WQMIX_Agents(QMIX_Agents):
    """The implementation of Weighted QMIX agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[MultiAgentBaseCallback] = None):
        super(WQMIX_Agents, self).__init__(config, envs, callback)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents)
        ff_mixer = QMIX_FF_mixer(dim_state, self.config.hidden_dim_ff_mix_net, self.n_agents)
        target_ff_mixer = QMIX_FF_mixer(dim_state, self.config.hidden_dim_ff_mix_net, self.n_agents)
        if self.config.policy == "Weighted_Mixing_Q_network":
            policy = REGISTRY_Policy["Weighted_Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=[mixer, mixer], ff_mixer=[ff_mixer, target_ff_mixer],
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"WQMIX currently does not support the policy named {self.config.policy}.")

        return policy
