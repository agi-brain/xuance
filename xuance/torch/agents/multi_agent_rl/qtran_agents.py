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


class QTRAN_Agents(MARLAgents):
    """The implementation of QTRAN agents.

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

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        if self.use_rnn:
            kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                          "dropout": config.dropout,
                          "rnn": config.rnn}
            representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        else:
            representation = REGISTRY_Representation[config.representation](*input_representation)
        mixer = VDN_mixer()
        if config.agent == "QTRAN_base":
            qtran_net = QTRAN_base(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                   config.n_agents, config.q_hidden_size[0]).to(device)
        elif config.agent == "QTRAN_alt":
            qtran_net = QTRAN_alt(config.dim_state[0], config.dim_act, config.qtran_net_hidden_dim,
                                  config.n_agents, config.q_hidden_size[0]).to(device)
        else:
            raise ValueError("Mixer {} not recognised.".format(config.agent))
        input_policy = get_policy_in_marl(config, representation, mixer, qtran_mixer=qtran_net)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_rnn=config.use_rnn,
                                                rnn=config.rnn)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        input_buffer = (config.n_agents, state_shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size, config.batch_size)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)

        learner = QTRAN_Learner(config, policy, optimizer, scheduler,
                                config.device, config.model_dir, config.gamma,
                                config.sync_frequency)
        super(QTRAN_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                           config.log_dir, config.model_dir)
        self.on_policy = False

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

    def train(self, i_step, n_epoch=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epoch):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
