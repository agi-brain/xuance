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


class VDAC_Agents(MARLAgents):
    """The implementation of VDAC agents.

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
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_rnn else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy
        if config.mixer == "VDN":
            mixer = VDN_mixer()
        elif config.mixer == "QMIX":
            mixer = QMIX_mixer(config.dim_state[0], config.hidden_dim_mixing_net, config.hidden_dim_hyper_net,
                               config.n_agents, device)
        elif config.mixer == "Independent":
            mixer = None
        else:
            raise AttributeError(f"Mixer named {config.mixer} is not defined!")
        input_policy = get_policy_in_marl(config, representation, mixer=mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_rnn=config.use_rnn,
                                                rnn=config.rnn,
                                                gain=config.gain)
        optimizer = torch.optim.Adam(policy.parameters(),
                                     lr=config.learning_rate, eps=1e-5,
                                     weight_decay=config.weight_decay)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = VDAC_Learner(config, policy, optimizer, None, config.device, config.model_dir, config.gamma)
        super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False
        self.on_policy = True

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
                    if self.use_rnn:
                        info_train = self.learner.update_recurrent(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
