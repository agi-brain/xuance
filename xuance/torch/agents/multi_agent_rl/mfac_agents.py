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


class MFAC_Agents(MARLAgents):
    """The implementation of Mean-Field AC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        device: the calculating device of the model, such as CPU or GPU.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_size = config.buffer_size
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        input_policy = get_policy_in_marl(config, representation)
        policy = REGISTRY_Policy[config.policy](*input_policy, gain=config.gain)
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
        memory = MeanField_OnPolicyBuffer(config.n_agents,
                                          state_shape,
                                          config.obs_shape,
                                          config.act_shape,
                                          config.rew_shape,
                                          config.done_shape,
                                          envs.num_envs,
                                          config.buffer_size,
                                          config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda,
                                          prob_space=config.act_prob_shape)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
        learner = MFAC_Learner(config, policy, optimizer, scheduler,
                               config.device, config.model_dir, config.gamma)
        super(MFAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.on_policy = True

    def act(self, obs_n, test_mode, act_mean=None, agent_mask=None):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_n = torch.Tensor(obs_n).to(self.device)

        _, dists = self.policy(obs_n, agents_id)
        acts = dists.stochastic_sample()

        n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
        action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
        act_neighbor_onehot = self.learner.onehot_action(acts, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
        act_mean_current = act_mean_current.cpu().detach().numpy()

        return acts.detach().cpu().numpy(), act_mean_current

    def values(self, obs, actions_mean):
        batch_size = len(obs)
        obs = torch.Tensor(obs).to(self.device)
        actions_mean = torch.Tensor(actions_mean).to(self.device)
        actions_mean = actions_mean.unsqueeze(1).expand(-1, self.n_agents, -1)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        values_n = self.policy.critic(obs, actions_mean, agents_id)
        hidden_states = None
        return hidden_states, values_n.detach().cpu().numpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epoch):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
