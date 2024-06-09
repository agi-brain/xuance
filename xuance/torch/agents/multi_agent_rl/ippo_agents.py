import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
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
                 envs: DummyVecMultiAgentEnv):
        super(IPPO_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.continuous_control = False
        self.n_epoch = config.n_epoch
        self.n_minibatch = config.n_minibatch
        self.use_global_state = config.use_global_state
        # create policy, optimizer, and lr_scheduler.
        self.policy = self._build_policy()
        optimizer = torch.optim.Adam(self.policy.parameters_model,
                                     lr=config.learning_rate, eps=1e-5,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                      start_factor=1.0, end_factor=config.end_factor_lr_decay,
                                                      total_iters=self.config.running_steps)
        # create experience replay buffer
        n_actions = None if self.continuous_control else {k: self.action_space[k].n for k in self.agent_keys}
        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        self.memory = buffer(agent_keys=self.agent_keys,
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
                             max_episode_steps=envs.max_episode_steps)
        self.buffer_size = self.memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
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
        agent = self.config.agent
        # build representations
        representation_actor = self._build_representation(self.config.representation, self.config)
        representation_critic = self._build_representation(self.config.representation, self.config)
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
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return IPPO_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            log_pi_a (dict): The log of pi.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            values_dict (dict): Critic values for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'log_pi_old': log_pi_a,
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys},
        self.memory.store(**experience_data)

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_size = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(batch_size) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(batch_size) for k in self.model_keys}
        return rnn_hidden_actor, rnn_hidden_critic

    def init_hidden_item(self,
                         i_env: int,
                         rnn_hidden_actor: Optional[dict] = None,
                         rnn_hidden_critic: Optional[dict] = None):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
            rnn_hidden_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            batch_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
        else:
            batch_index = i_env
        for k in self.model_keys:
            rnn_hidden_actor[k] = self.policy.actor_representation[k].init_hidden_item(batch_index, *rnn_hidden_actor[k])
            rnn_hidden_critic[k] = self.policy.critic_representation[k].init_hidden_item(batch_index, *rnn_hidden_critic[k])
        return rnn_hidden_actor, rnn_hidden_critic

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
            actions_out = pi_dists[key].stochastic_sample()
            log_pi_a = pi_dists[key].log_prob(actions_out).cpu().detach().numpy()
            if self.use_rnn:
                if self.continuous_control:
                    actions_out = actions_out.reshape(n_env, self.n_agents, -1)
                else:
                    actions_out = actions_out.reshape(n_env, self.n_agents)
                log_pi_a = log_pi_a.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            log_pi_a_dict = {k: log_pi_a[:, i] for i, k in enumerate(self.agent_keys)}
            if not test_mode:
                rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                           agent_ids=agents_id,
                                                                           rnn_hidden=rnn_hidden_critic)
                values_dict = {k: values_out[key][:, i].reshape(n_env).cpu().detach().numpy()
                               for i, k in enumerate(self.agent_keys)}
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

            actions_out = {k: pi_dists[k].stochastic_sample() for k in self.agent_keys}
            log_pi_a = {k: pi_dists[k].log_prob(actions_out[k]).cpu().detach().numpy() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_out[k].cpu().detach().numpy()[e].reshape([-1]) for k in self.agent_keys}
                                for e in range(n_env)]
            else:
                actions_dict = [{k: actions_out[k].cpu().detach().numpy()[e].reshape([]) for k in self.agent_keys}
                                for e in range(n_env)]
            log_pi_a_dict = {k: log_pi_a[k].reshape([n_env]) for i, k in enumerate(self.agent_keys)}

            if not test_mode:
                rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                           rnn_hidden=rnn_hidden_critic)
                values_dict = {k: values_out[k].cpu().detach().numpy().reshape([n_env]) for k in self.agent_keys}

        return rnn_hidden_actor_new, rnn_hidden_critic_new, actions_dict, log_pi_a_dict, values_dict

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    rnn_hidden_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
        """
        n_env = 1
        rnn_hidden_critic_i = None
        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_rnn:
                hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
                rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                    hidden_item_index, *rnn_hidden_critic[key])}
                batch_size = n_env * self.n_agents
                obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                obs_input = {key: obs_array.reshape([batch_size, 1, -1])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(
                    self.device)
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(obs_dict)])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_out = values_out[key].reshape(self.n_agents)
            values_dict = {k: values_out[i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}

        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    i_env, *rnn_hidden_critic[k]) for k in self.agent_keys}
            obs_input = {k: obs_dict[k][None, :] for k in self.agent_keys} if self.use_rnn else obs_dict

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       rnn_hidden=rnn_hidden_critic_i)
            values_dict = {k: values_out[k].cpu().detach().numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict

    def train_epochs(self, n_epochs=1):
        """
        Train the model for numerous epochs.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        indexes = np.arange(self.buffer_size)
        for _ in range(n_epochs):
            np.random.shuffle(indexes)
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                sample_idx = indexes[start:end]
                sample = self.memory.sample(sample_idx)
                info_train = self.learner.update_rnn(sample) if self.use_rnn else self.learner.update(sample)
        self.memory.clear()
        return info_train

    def run_episode(self, n_episodes: int):
        """
        Run a number of episodes to collect data sequence.
        Note: This method is called when self.use_rnn is True.

        Parameters:
            n_episodes: The number of episodes to run.
        """
        obs_dict, info = self.envs.reset()
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state if self.use_global_state else None
        episode_count = 0
        self.memory.clear_episodes()
        rnn_hidden_actor, rnn_hidden_critic = self.init_rnn_hidden(self.n_envs)
        while episode_count < n_episodes:
            step_info = {}
            rnn_hidden_next_actor, rnn_hidden_next_critic, actions_dict, log_pi_a_dict, values_dict = self.action(
                obs_dict=obs_dict, avail_actions_dict=avail_actions,
                rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic, test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info, **{'state': state})
            obs_dict = deepcopy(next_obs_dict)
            avail_actions = deepcopy(next_avail_actions)
            rnn_hidden_actor = deepcopy(rnn_hidden_next_actor)
            rnn_hidden_critic = deepcopy(rnn_hidden_next_critic)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    self.current_step += info[i]["episode_step"]
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                         rnn_hidden_critic=rnn_hidden_critic)
                    self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor, rnn_hidden_critic)
                    obs_dict[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episode(self.n_envs)
                    if self.memory.full:
                        train_info = self.train_epochs(n_epochs=self.n_epoch)
                        self.log_infos(train_info, self.current_step)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
            return

        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            step_info = {}
            _, _, actions_dict, log_pi_a_dict, values_dict = self.action(obs_dict=obs_dict,
                                                                         avail_actions_dict=avail_actions,
                                                                         test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info, **{'state': state})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        _, value_next = self.values_next(i_env=i, obs_dict=next_obs_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                train_info = self.train_epochs(n_epochs=self.n_epoch)
                self.log_infos(train_info, self.current_step)
            obs_dict, avail_actions, state = deepcopy(next_obs_dict), deepcopy(next_avail_actions), deepcopy(next_state)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs

    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
            scores (List(float)): A list of cumulative rewards for each episode.
        """
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = test_envs.reset()
        avail_actions = test_envs.buf_avail_actions if self.use_actions_mask else None
        if self.use_rnn:
            rnn_hidden_actor, _ = self.init_rnn_hidden(num_envs)
        else:
            rnn_hidden_actor = None
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < n_episodes:
            rnn_hidden_actor, _, actions_dict, _, _ = self.action(obs_dict=obs_dict,
                                                                  avail_actions_dict=avail_actions,
                                                                  rnn_hidden_actor=rnn_hidden_actor,
                                                                  test_mode=True)
            obs_dict, rewards_dict, terminated_dict, truncated, info = test_envs.step(actions_dict)
            avail_actions = test_envs.buf_avail_actions if self.use_actions_mask else None
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if self.use_rnn:
                        for key in self.model_keys:
                            rnn_hidden_actor[key] = self.policy.actor_representation[key].init_hidden_item(
                                i, *rnn_hidden_actor[key])
                    obs_dict[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    current_episode += 1
                    if best_score < episode_score:
                        best_score = episode_score
                        episode_videos = videos[i].copy()
                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, episode_score))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores),
        }
        self.log_infos(test_info, self.current_step)
        test_envs.close()
        return scores
