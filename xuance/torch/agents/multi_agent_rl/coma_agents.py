import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import List, Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import COMA_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import COMA_Buffer, COMA_Buffer_RNN


class COMA_Agents(MARLAgents):
    """The implementation of COMA agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        device: the calculating device of the model, such as CPU or GPU.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(COMA_Agents, self).__init__(config, envs)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        self.use_global_state = config.use_global_state

        # create policy, optimizer, and lr_scheduler.
        self.policy = self._build_policy()
        optimizer = [torch.optim.Adam(self.policy.parameters_actor, config.learning_rate_actor, eps=1e-5),
                     torch.optim.Adam(self.policy.parameters_critic, config.learning_rate_critic, eps=1e-5)]
        scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer[0], start_factor=1.0, end_factor=0.5,
                                                       total_iters=self.config.running_steps),
                     torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1.0, end_factor=0.5,
                                                       total_iters=self.config.running_steps)]

        # create experience replay buffer
        buffer = COMA_Buffer_RNN if self.use_rnn else COMA_Buffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps,
                        dim_act=config.dim_act, td_lambda=config.td_lambda)

        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        # initialize the hidden states of the RNN is use RNN-based representations.
        self.rnn_hidden_actor, self.rnn_hidden_critic = self.init_rnn_hidden(self.n_envs)

        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, envs.max_episode_steps,
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
        representation_actor = self._build_representation(self.config.representation, self.config)
        representation_critic = self._build_representation(self.config.representation_critic, self.config)

        # build policies
        if self.config.policy == "Categorical_COMA_Policy":
            policy = REGISTRY_Policy["Categorical_COMA_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=representation_actor, representation_critic=representation_critic,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                use_global_state=self.use_global_state)
            self.continuous_control = False
        else:
            raise AttributeError(f"COMA currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return COMA_Learner(*args)

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
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        n_env = len(obs_dict)
        avail_actions_input = None
        rnn_hidden_critic_new, values_dict = {}, {}

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_rnn:
                batch_size = n_env * self.n_agents
                obs_array = np.array([itemgetter(*self.agent_keys)(data) for data in obs_dict])
                obs_input = {key: obs_array.reshape([batch_size, 1, -1])}
                if self.use_actions_mask:
                    avail_actions_array = np.array([itemgetter(*self.agent_keys)(data) for data in avail_actions_dict])
                    avail_actions_input = {key: avail_actions_array.reshape([batch_size, 1, -1])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(
                    self.device)
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])}
                if self.use_actions_mask:
                    avail_actions_input = {
                        key: np.array([itemgetter(*self.agent_keys)(data) for data in avail_actions_dict])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            rnn_hidden_actor_new, pi_dists = self.policy(observation=obs_input,
                                                         agent_ids=agents_id,
                                                         avail_actions=avail_actions_input,
                                                         rnn_hidden=rnn_hidden_actor)
            if not test_mode:
                rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                           agent_ids=agents_id,
                                                                           rnn_hidden=rnn_hidden_critic)
                values_dict = [{k: values_out[key][e, i, 0].cpu().detach().numpy()
                                for i, k in enumerate(self.agent_keys)} for e in range(n_env)]

            actions_out = pi_dists[key].stochastic_sample()
            log_pi_a = pi_dists[key].log_prob(actions_out).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            log_pi_a_dict = [{k: log_pi_a[e, i] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]
        else:
            if self.use_rnn:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict])[:, None] for k in
                             self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict])[:, None]
                                           for k in self.agent_keys}
            else:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict]) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict]) for k in
                                           self.agent_keys}

            rnn_hidden_actor_new, pi_dists = self.policy(observation=obs_input,
                                                         avail_actions=avail_actions_input,
                                                         rnn_hidden=rnn_hidden_actor)
            if not test_mode:
                rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                           rnn_hidden=rnn_hidden_critic)
                values_dict = [{k: values_out[k][e, 0].cpu().detach().numpy() for k in self.agent_keys}
                               for e in range(n_env)]

            actions_out, log_pi_a = {}, {}
            for key in self.agent_keys:
                if self.use_rnn:
                    actions_out[key] = pi_dists[key].stochastic_sample().squeeze(1)
                    log_pi_a[key] = pi_dists[key].log_prob(actions_out[key]).cpu().detach().numpy()
                else:
                    actions_out[key] = pi_dists[key].stochastic_sample()
                    log_pi_a[key] = pi_dists[key].log_prob(actions_out[key]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[k].cpu().detach().numpy()[e] for k in self.agent_keys}
                            for e in range(n_env)]
            log_pi_a_dict = [{k: log_pi_a[k][e] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]

        return rnn_hidden_actor_new, rnn_hidden_critic_new, actions_dict, log_pi_a_dict, values_dict

    def values_next(self,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None):
        batch_size = len(obs_n)
        # build critic input
        obs_n = torch.Tensor(obs_n).to(self.device)
        actions_n = torch.Tensor(actions_n).unsqueeze(-1).to(self.device)
        actions_in = torch.Tensor(actions_onehot).unsqueeze(1).to(self.device)
        actions_in = actions_in.view(batch_size, 1, -1).repeat(1, self.n_agents, 1)
        agent_mask = 1 - torch.eye(self.n_agents, device=self.device)
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.dim_act).view(self.n_agents, -1)
        actions_in = actions_in * agent_mask.unsqueeze(0)
        if self.use_global_state:
            state = torch.Tensor(state).unsqueeze(1).to(self.device).repeat(1, self.n_agents, 1)
            critic_in = torch.concat([state, obs_n, actions_in], dim=-1)
        else:
            critic_in = torch.concat([obs_n, actions_in], dim=-1)
        # get critic values
        hidden_state, values_n = self.policy.get_values(critic_in, target=True)

        target_values = values_n.gather(-1, actions_n.long())
        return hidden_state, target_values.detach().cpu().numpy()

    def train(self, i_step, **kwargs):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
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
                        info_train = self.learner.update_recurrent(sample, self.egreedy)
                    else:
                        info_train = self.learner.update(sample, self.egreedy)
            self.memory.clear()
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
