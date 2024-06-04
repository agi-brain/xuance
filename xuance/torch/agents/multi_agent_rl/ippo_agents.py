import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from xuance.environment import DummyVecMultiAgentEnv
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
                 envs: DummyVecMultiAgentEnv):
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
                            max_episode_steps=envs.max_episode_steps)
        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        self.memory = buffer(**input_buffer)

        self.buffer_size = self.memory.buffer_size
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
            values_dict (List[dict]): Critic values for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([itemgetter(k)(data) for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([itemgetter(k)(data) for data in actions_dict]) for k in self.agent_keys},
            'log_pi_old': {k: np.array([itemgetter(k)(data) for data in log_pi_a]) for k in self.agent_keys},
            'rewards': {k: np.array([itemgetter(k)(data) for data in rewards_dict]) for k in self.agent_keys},
            'values': {k: np.array([itemgetter(k)(data) for data in values_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([itemgetter(k)(data) for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([itemgetter(k)(data['agent_mask']) for data in info])
                           for k in self.agent_keys},
            'avail_actions': {k: np.array([itemgetter(k)(data) for data in avail_actions]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        self.memory.store(**experience_data)

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
                    rnn_hidden_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
        """
        n_env = 1

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_rnn:
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
                                                                       rnn_hidden=rnn_hidden_critic)
            values_dict = {k: values_out[key][0, i, 0].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}

        else:
            obs_input = {k: obs_dict[k][:, None] for k in self.agent_keys} if self.use_rnn else obs_dict

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=obs_input,
                                                                       rnn_hidden=rnn_hidden_critic)
            values_dict = {k: values_out[k][0].cpu().detach().numpy() for k in self.agent_keys}

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
        for _ in tqdm(range(n_steps)):
            step_info = {}
            rnn_hidden_next_actor, rnn_hidden_next_critic, actions_dict, log_pi_a_dict, values_dict = self.action(
                obs_dict=obs_dict, avail_actions_dict=avail_actions,
                rnn_hidden_actor=self.rnn_hidden_actor, rnn_hidden_critic=self.rnn_hidden_critic, test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state
            next_avail_actions = self.envs.buf_avail_actions
            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info, **{'state': state})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        _, value_next = self.values_next(obs_dict=next_obs_dict[i], rnn_hidden_critic=rnn_hidden_next_critic)
                    self.memory.finish_path(i_env=i, value_next=value_next, value_normalizer=self.learner.value_normalizer)
                train_info = self.train_epochs(n_epochs=self.n_epoch)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)
            avail_actions = deepcopy(next_avail_actions)
            state = deepcopy(next_state)
            self.rnn_hidden_actor = deepcopy(rnn_hidden_next_actor)
            self.rnn_hidden_critic = deepcopy(rnn_hidden_next_critic)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        _, value_next = self.values_next(obs_dict=obs_dict[i], rnn_hidden_critic=self.rnn_hidden_critic)
                    if self.use_rnn:
                        self.init_hidden_item(i)
                        raise NotImplementedError
                    else:
                        self.memory.finish_path(i_env=i, value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
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
        avail_actions = test_envs.buf_avail_actions
        rnn_hidden_actor = self.init_rnn_hidden(num_envs)
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < n_episodes:
            rnn_hidden_next_actor, _, actions_dict, _, _ = self.action(obs_dict=obs_dict,
                                                                       avail_actions_dict=avail_actions,
                                                                       rnn_hidden_actor=rnn_hidden_actor,
                                                                       test_mode=True)
            obs_dict, rewards_dict, terminated_dict, truncated, info = test_envs.step(actions_dict)
            avail_actions = test_envs.buf_avail_actions
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            rnn_hidden_actor = deepcopy(rnn_hidden_next_actor)
            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if self.use_rnn:
                        for key in self.model_keys:
                            rnn_hidden_actor[key] = self.policy.actor_representation[key].init_hidden_item(
                                i, *rnn_hidden_actor[key])
                    obs_dict[i] = info[i]["reset_obs"]
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
