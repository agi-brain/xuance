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


class MFQ_Agents(MARLAgents):
    """The implementation of Mean-Field Q agents.

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
        self.use_rnn, self.rnn = config.use_rnn, config.rnn
        self.rnn_hidden = None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy)
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
        memory = MeanField_OffPolicyBuffer(config.n_agents,
                                           state_shape,
                                           config.obs_shape,
                                           config.act_shape,
                                           config.act_prob_shape,
                                           config.rew_shape,
                                           config.done_shape,
                                           envs.num_envs,
                                           config.buffer_size,
                                           config.batch_size)
        learner = MFQ_Learner(config, policy, optimizer, scheduler,
                              config.device, config.model_dir, config.gamma,
                              config.sync_frequency)
        super(MFQ_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                         config.log_dir, config.model_dir)
        self.on_policy = False

    def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).to(self.device)
        act_mean = torch.Tensor(act_mean).unsqueeze(dim=-2).repeat(1, self.n_agents, 1).to(self.device)

        if self.use_rnn:  # awaiting to be tested
            batch_agents = batch_size * self.n_agents
            hidden_state, greedy_actions, q_output = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                 act_mean.view(batch_agents, 1, -1),
                                                                 agents_id.view(batch_agents, 1, -1),
                                                                 *rnn_hidden,
                                                                 avail_actions=avail_actions)
        else:
            hidden_state, greedy_actions, q_output = self.policy(obs_in, act_mean, agents_id)
        n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
        action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
        act_neighbor_sample = self.policy.sample_actions(logits=q_output).to(self.device)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
        act_mean_current = act_mean_current.cpu().detach().numpy()
        greedy_actions = greedy_actions.cpu().detach().numpy()
        if test_mode:
            return hidden_state, greedy_actions, act_mean_current
        else:
            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions, act_mean_current
            else:
                return hidden_state, greedy_actions, act_mean_current

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
