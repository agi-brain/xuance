import torch
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, QTRAN_base, QTRAN_alt, VDN_mixer
from xuance.torch.agents import OffPolicyMARLAgents


class QTRAN_Agents(OffPolicyMARLAgents):
    """The implementation of QTRAN agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(QTRAN_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.use_global_state = True

        # build policy, optimizers, schedulers
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

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        dim_action = self.action_space
        mixer = VDN_mixer()
        if self.config.agent == "QTRAN_base":
            qtran_net = QTRAN_base(dim_state, self.config.dim_act, self.config.qtran_net_hidden_dim,
                                   self.config.n_agents, self.config.q_hidden_size[0], device)
        elif self.config.agent == "QTRAN_alt":
            qtran_net = QTRAN_alt(dim_state, self.config.dim_act, self.config.qtran_net_hidden_dim,
                                  self.config.n_agents, self.config.q_hidden_size[0], device)
        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        if self.config.policy == "Qtran_Mixing_Q_network":
            policy = REGISTRY_Policy["Qtran_Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, qtran_net=qtran_net,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"QMIX currently does not support the policy named {self.config.policy}.")

        return policy

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, _, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                             agents_id.view(batch_agents, 1, -1),
                                                             *rnn_hidden,
                                                             avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            greedy_actions = greedy_actions.view(batch_size, self.n_agents)
        else:
            hidden_state, _, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
        greedy_actions = greedy_actions.cpu().detach().numpy()

        if test_mode:
            return hidden_state, greedy_actions
        else:
            if avail_actions is None:
                random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            else:
                random_actions = Categorical(torch.Tensor(avail_actions)).sample().numpy()
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions
            else:
                return hidden_state, greedy_actions

    def train(self, i_step, n_epochs=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epochs):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
