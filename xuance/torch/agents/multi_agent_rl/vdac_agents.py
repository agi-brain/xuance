import torch
import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, VDN_mixer, QMIX_mixer
from xuance.torch.agents import OnPolicyMARLAgents


class VDAC_Agents(OnPolicyMARLAgents):
    """The implementation of VDAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(VDAC_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.mixer = config.mixer

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent
        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        # create mixer
        if self.mixer == "VDN":
            mixer = VDN_mixer()
        elif self.mixer == "QMIX":
            dim_state = self.state_space.shape[-1]
            mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net, self.config.hidden_dim_hyper_net,
                               self.n_agents, self.device)
        elif self.mixer == "Independent":
            mixer = None
        else:
            raise AttributeError(f"Mixer named {self.mixer} is not supported in XuanCe!")
        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy


    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).reshape([batch_size, self.n_agents, -1]).to(self.device)
        if state is not None:
            state = torch.Tensor(state).to(self.device)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions).to(self.device)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, dists, values_tot = self.policy(obs_in.reshape(batch_agents, 1, -1),
                                                          agents_id.reshape(batch_agents, 1, -1),
                                                          *rnn_hidden,
                                                          avail_actions=avail_actions.reshape(batch_agents, 1, -1),
                                                          state=state.unsqueeze(2))
            actions = dists.stochastic_sample()
            actions = actions.reshape(batch_size, self.n_agents)
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        else:
            hidden_state, dists, values_tot = self.policy(obs_in, agents_id,
                                                          avail_actions=avail_actions,
                                                          state=state)
            actions = dists.stochastic_sample()
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        return hidden_state, actions.detach().cpu().numpy(), values_tot.detach().cpu().numpy()
