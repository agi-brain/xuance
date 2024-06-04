import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy, QMIX_mixer
from xuance.torch.learners import QMIX_Learner
from xuance.torch.agents import MARLAgents
from xuance.torch.agents.multi_agent_rl.iql_agents import IQL_Agents
from xuance.common import MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN


class DCG_Agents(MARLAgents):
    """The implementation of DCG agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        device: the calculating device of the model, such as CPU or GPU.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        self.gamma = config.gamma
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        if self.use_rnn:
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                          "dropout": config.dropout,
                          "rnn": config.rnn}
            representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        else:
            representation = REGISTRY_Representation[config.representation](*input_representation)
        repre_state_dim = representation.output_shapes['state'][0]
        from xuance.torch.policies.coordination_graph import DCG_utility, DCG_payoff, Coordination_Graph
        utility = DCG_utility(repre_state_dim, config.hidden_utility_dim, config.dim_act).to(device)
        payoffs = DCG_payoff(repre_state_dim * 2, config.hidden_payoff_dim, config.dim_act, config).to(device)
        dcgraph = Coordination_Graph(config.n_agents, config.graph_type)
        dcgraph.set_coordination_graph(device)
        if config.env_name == "StarCraft2":
            action_space = config.action_space
        else:
            action_space = config.action_space[config.agent_keys[0]]
        if config.agent == "DCG_S":
            policy = REGISTRY_Policy[config.policy](action_space,
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, config.hidden_bias_dim,
                                                    None, None, torch.nn.ReLU, device,
                                                    use_rnn=config.use_rnn,
                                                    rnn=config.rnn)
        else:
            policy = REGISTRY_Policy[config.policy](action_space,
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, None,
                                                    None, None, torch.nn.ReLU, device,
                                                    use_rnn=config.use_rnn,
                                                    rnn=config.rnn)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)

        from xuance.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
        learner = DCG_Learner(config, policy, optimizer, scheduler,
                              config.device, config.model_dir, config.gamma,
                              config.sync_frequency)
        super(DCG_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                         config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, *rnn_hidden, avail_actions=None, test_mode=False):
        batch_size = obs_n.shape[0]
        obs_n = torch.Tensor(obs_n).to(self.device)
        with torch.no_grad():
            obs_in = obs_n.view(batch_size * self.n_agents, 1, -1)
            rnn_hidden_next, hidden_states = self.learner.get_hidden_states(obs_in, *rnn_hidden)
            greedy_actions = self.learner.act(hidden_states.view(batch_size, self.n_agents, -1),
                                              avail_actions=avail_actions)
        greedy_actions = greedy_actions.cpu().detach().numpy()

        if test_mode:
            return rnn_hidden_next, greedy_actions
        else:
            if avail_actions is None:
                random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            else:
                random_actions = Categorical(torch.Tensor(avail_actions)).sample().numpy()
            if np.random.rand() < self.egreedy:
                return rnn_hidden_next, random_actions
            else:
                return rnn_hidden_next, greedy_actions

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample = self.memory.sample()
                if self.use_rnn:
                    info_train = self.learner.update_recurrent(sample)
                else:
                    info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
