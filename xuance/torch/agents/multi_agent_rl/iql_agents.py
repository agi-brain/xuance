import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from xuance.environment import DummyVecMutliAgentEnv
from xuance.torch import Tensor
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import IQL_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OffPolicyBuffer_Share, MARL_OffPolicyBuffer_Split, MARL_OffPolicyBuffer_RNN


class IQL_Agents(MARLAgents):
    """The implementation of Independent Q-Networks agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        super(IQL_Agents, self).__init__(config, envs)

        self.use_recurrent = config.use_recurrent
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.policy = self._build_policy()
        optimizer = torch.optim.Adam(self.policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=self.config.running_steps)

        # create experience replay buffer
        input_buffer = dict(n_agents=self.config.n_agents,
                            agent_keys=self.agent_keys,
                            state_space=None,
                            obs_space=self.observation_space[self.agent_keys[0]],
                            act_space=self.action_space[self.agent_keys[0]],
                            n_envs=self.n_envs,
                            buffer_size=self.config.buffer_size,
                            batch_size=self.config.batch_size,
                            max_episode_length=envs.max_episode_length)
        if self.use_parameter_sharing:
            buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer_Share
            self.memory = buffer(**input_buffer)
        else:
            input_buffer['obs_space'] = self.observation_space
            input_buffer['act_space'] = self.action_space
            buffer = MARL_OffPolicyBuffer_RNN if self.use_recurrent else MARL_OffPolicyBuffer_Share
            self.memory = buffer(**input_buffer)

        # create learner
        self.learner = self._build_learner(self.config, self.model_keys, self.policy, optimizer, scheduler)

    def _build_policy(self):
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
        representation = {key: None for key in self.model_keys}
        for key in self.model_keys:
            input_shape = self.observation_space[key].shape
            if self.config.representation == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif self.config.representation == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif self.config.representation == "Basic_RNN":
                raise NotImplementedError
            else:
                raise f"The IQL currently does not support the representation of {self.config.representation}."

        # build policies
        if self.config.policy == "Basic_Q_network_marl":
            policy = REGISTRY_Policy["Basic_Q_network_marl"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys)
        else:
            raise f"The IQL currently does not support the policy named {self.config.policy}."

        return policy

    def _build_learner(self, config, model_keys, policy, optimizer, scheduler):
        return IQL_Learner(config, model_keys, policy, optimizer, scheduler)

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if self.use_recurrent:
            batch_agents = batch_size * self.n_agents
            hidden_state, greedy_actions, _ = self.policy(obs_in.view(batch_agents, 1, -1),
                                                          agents_id.view(batch_agents, 1, -1),
                                                          *rnn_hidden,
                                                          avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            greedy_actions = greedy_actions.view(batch_size, self.n_agents)
        else:
            hidden_state, greedy_actions, _ = self.policy(obs_in, agents_id, avail_actions=avail_actions)
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

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample = self.memory.sample()
                if self.use_recurrent:
                    info_train = self.learner.update_recurrent(sample)
                else:
                    info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
