import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from xuance.environment import DummyVecMutliAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import MAPPO_Clip_Learner
from xuance.torch.agents import MARLAgents
from xuance.torch.agents.multi_agent_rl.ippo_agents import IPPO_Agents
from xuance.common import MARL_OnPolicyBuffer, MARL_OnPolicyBuffer_RNN


class MAPPO_Agents(IPPO_Agents):
    """The implementation of MAPPO agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        super(MAPPO_Agents, self).__init__(config, envs)
        self.share_values = True if config.rew_shape[0] == 1 else False

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
        representation_actor = {key: None for key in self.model_keys}
        representation_critic = {key: None for key in self.model_keys}
        for key in self.model_keys:
            input_shape = self.observation_space[key].shape
            if self.config.representation == "Basic_Identical":
                representation_actor[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                       device=self.device)
                representation_critic[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                        device=self.device)
            elif self.config.representation == "Basic_MLP":
                representation_actor[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
                representation_critic[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif self.config.representation == "Basic_RNN":
                representation_actor[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
                representation_critic[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
            else:
                raise AttributeError(f"IPPO currently doesn't support the {self.config.representation} representation.")

        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=representation_actor, representation_critic=representation_critic,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=representation_actor, representation_critic=representation_critic,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"IPPO currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return MAPPO_Clip_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).reshape([batch_size, self.n_agents, -1]).to(self.device)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions).to(self.device)
        if self.use_rnn:  # use recurrent networks
            batch_agents = batch_size * self.n_agents
            hidden_state, dists = self.policy(obs_in.reshape(batch_agents, 1, -1),
                                              agents_id.reshape(batch_agents, 1, -1),
                                              *rnn_hidden,
                                              avail_actions=avail_actions.reshape(batch_agents, 1, -1))
            actions = dists.stochastic_sample()
            log_pi_a = dists.log_prob(actions).reshape(batch_size, self.n_agents)
            actions = actions.reshape(batch_size, self.n_agents)
        else:
            hidden_state, dists = self.policy(obs_in, agents_id, avail_actions=avail_actions)
            actions = dists.stochastic_sample()
            log_pi_a = dists.log_prob(actions)
        return hidden_state, actions.detach().cpu().numpy(), log_pi_a.detach().cpu().numpy()

    def values(self, obs_n, *rnn_hidden, state=None):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        # build critic input
        if self.use_global_state:
            state = torch.Tensor(state).unsqueeze(1).to(self.device)
            obs_n = torch.Tensor(obs_n).reshape([batch_size, 1, -1]).to(self.device)
            critic_in = torch.concat([obs_n.expand(-1, self.n_agents, -1),
                                      state.expand(-1, self.n_agents, -1)], dim=-1)
        else:
            critic_in = torch.Tensor(obs_n).reshape([batch_size, 1, -1]).to(self.device)
            critic_in = critic_in.expand(-1, self.n_agents, -1)
        # get critic values
        if self.use_rnn:
            hidden_state, values_n = self.policy.get_values(critic_in.unsqueeze(2),  # add a sequence length axis.
                                                            agents_id.unsqueeze(2),
                                                            *rnn_hidden)
            values_n = values_n.squeeze(2)
        else:
            hidden_state, values_n = self.policy.get_values(critic_in, agents_id)

        return hidden_state, values_n.detach().cpu().numpy()

    def train(self, i_step, **kwargs):
        info_train = {}
        if self.memory.full:
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
