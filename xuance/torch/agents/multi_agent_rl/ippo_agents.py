import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from torch.distributions import Categorical
from xuance.environment import DummyVecMutliAgentEnv
from xuance.torch import Tensor
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import IPPO_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OnPolicyBuffer, MARL_OnPolicyBuffer_RNN


class IPPO_Agents(MARLAgents):
    """The implementation of Independent PPO agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        super(IPPO_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.continuous_control = False

        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        self.use_global_state = config.use_global_state

        # create representation for actor
        self.policy = self._build_policy()
        optimizer = torch.optim.Adam(self.policy.parameters_model,
                                     lr=config.learning_rate, eps=1e-5,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=self.config.running_steps)

        # create experience replay buffer
        n_actions = {k: self.action_space[k].n for k in self.agent_keys} if self.continuous_control else None
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space if self.use_global_state else None,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.config.buffer_size,
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.config.gamma,
                            gae_lam=self.config.gae_lambda,
                            n_actions=n_actions,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_length=envs.max_episode_length)
        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        self.memory = buffer(**input_buffer)

        self.buffer_size = self.memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        # initialize the hidden states of the RNN is use RNN-based representations.
        self.rnn_hidden_actor, self.rnn_hidden_critic = self.init_rnn_hidden(self.n_envs)

        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, envs.max_episode_length,
                                           self.policy, optimizer, scheduler)

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
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"IPPO currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return IPPO_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.
        """
        rnn_hidden_actor, rnn_hidden_critic = {}, {}
        for key in self.model_keys:
            if self.use_rnn:
                batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
                rnn_hidden_actor[key] = self.policy.actor_representation[key].init_hidden(batch)
                rnn_hidden_critic[key] = self.policy.critic_representation[key].init_hidden(batch)
            else:
                rnn_hidden_actor[key] = [None, None]
                rnn_hidden_critic[key] = [None, None]
        return rnn_hidden_actor, rnn_hidden_critic

    def init_hidden_item(self, i_env):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        for key in self.model_keys:
            self.rnn_hidden_actor[key] = self.policy.actor_representation[key].init_hidden_item(
                i_env, *self.rnn_hidden_actor[key])
            self.rnn_hidden_critic[key] = self.policy.critic_representation[key].init_hidden_item(
                i_env, *self.rnn_hidden_critic[key])

    def action(self,
               obs_dict: Optional[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
        """
        n_env = len(obs_dict)
        avail_actions_dict = None

        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).view([batch_size, self.n_agents, -1]).to(self.device)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions).to(self.device)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, dists = self.policy(obs_in.view(batch_agents, 1, -1),
                                              agents_id.view(batch_agents, 1, -1),
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
            critic_in = state.expand(-1, self.n_agents, -1)
        else:
            critic_in = torch.Tensor(obs_n).to(self.device)
        # get critic values
        if self.use_rnn:
            hidden_state, values_n = self.policy.get_values(critic_in.unsqueeze(2),  # add a sequence length axis.
                                                            agents_id.unsqueeze(2),
                                                            *rnn_hidden)
            values_n = values_n.squeeze(2)
        else:
            hidden_state, values_n = self.policy.get_values(critic_in, agents_id)

        return hidden_state, values_n.detach().cpu().numpy()

    def train_epochs(self, n_epochs=1):
        """
        Train the model for numerous epochs.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_rnn(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.memory.clear()
        return info_train

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions
        state = self.envs.buf_state if self.use_global_state else None
        for i_step in tqdm(range(n_steps)):
            step_info = {}
            rnn_hidden_next, actions_dict = self.action(obs_dict=obs_dict,
                                                        state=state,
                                                        avail_actions_dict=avail_actions,
                                                        rnn_hidden=self.rnn_hidden_actor,
                                                        test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state
            next_avail_actions = self.envs.buf_avail_actions
            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state})
            train_info = self.train_epochs(n_epochs=self.n_epoch)
            self.log_infos(train_info, self.c)




    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
            scores (List(float)): A list of cumulative rewards for each episode.
        """
        raise NotImplementedError

