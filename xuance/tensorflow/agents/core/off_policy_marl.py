from tqdm import tqdm
import numpy as np
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from xuance.common import Optional, List, Union, MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN
from xuance.environment import DummyVecMultiAgentEnv
from xuance.tensorflow import Tensor, Module
from xuance.tensorflow.utils.distributions import Categorical
from xuance.tensorflow.agents.base import MARLAgents


class OffPolicyMARLAgents(MARLAgents):
    """The core class for on-policy algorithm with single agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(OffPolicyMARLAgents, self).__init__(config, envs)
        self.start_greedy = config.start_greedy if hasattr(config, "start_greedy") else None
        self.end_greedy = config.end_greedy if hasattr(config, "start_greedy") else None
        self.delta_egreedy: Optional[float] = None
        self.e_greedy: Optional[float] = None

        self.start_noise = config.start_noise if hasattr(config, "start_noise") else None
        self.end_noise = config.end_noise if hasattr(config, "end_noise") else None
        self.delta_noise: Optional[float] = None
        self.noise_scale: Optional[float] = None
        self.actions_low = self.action_space.low if hasattr(self.action_space, "low") else None
        self.actions_high = self.action_space.high if hasattr(self.action_space, "high") else None

        self.auxiliary_info_shape = None
        self.memory: Optional[MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN] = None

    def _build_memory(self):
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space if self.use_global_state else None,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.config.buffer_size,
                            batch_size=self.config.batch_size,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length)
        Buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict, avail_actions_next,
                         rewards_dict, terminals_dict, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            obs_next_dict (List[dict]): Next observations for each agent in self.agent_keys.
            avail_actions_next (List[dict]): The next actions mask values for each agent in self.agent_keys.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'obs_next': {k: np.array([data[k] for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
            experience_data['state_next'] = np.array(kwargs['next_state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
            experience_data['avail_actions_next'] = {k: np.array([data[k] for data in avail_actions_next])
                                                     for k in self.agent_keys}
        self.memory.store(**experience_data)

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.

        Returns:
            rnn_hidden_states: The hidden states for RNN.
        """
        rnn_hidden_states = None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_states = {k: self.policy.representation[k].init_hidden(batch) for k in self.model_keys}
        return rnn_hidden_states

    def init_hidden_item(self, i_env: int,
                         rnn_hidden: Optional[dict] = None):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
            rnn_hidden (Optional[dict]): The RNN hidden states of actor representation.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            batch_index = list(range(i_env * self.n_agents, (i_env + 1) * self.n_agents))
        else:
            batch_index = [i_env, ]
        for key in self.model_keys:
            rnn_hidden[key] = self.policy.representation[key].init_hidden_item(batch_index, *rnn_hidden[key])
        return rnn_hidden

    def _update_explore_factor(self):
        if self.e_greedy is not None:
            if self.e_greedy > self.end_greedy:
                self.e_greedy = self.start_greedy - self.delta_egreedy * self.current_step
            else:
                self.e_greedy = self.end_greedy
        elif self.noise_scale is not None:
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.start_noise - self.delta_noise * self.current_step
            else:
                self.noise_scale = self.end_noise
        else:
            return

    def exploration(self, batch_size: int,
                    pi_actions_dict: Union[List[dict], dict],
                    avail_actions_dict: Optional[List[dict]] = None):
        """Returns the actions for exploration.

        Parameters:
            batch_size (int): The batch size.
            pi_actions_dict (Optional[List[dict], dict]): The original output actions.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.

        Returns:
            explore_actions: The actions with noisy values.
        """
        if self.e_greedy is not None:
            if np.random.rand() < self.e_greedy:
                if self.use_actions_mask:
                    explore_actions = [{k: Categorical(Tensor(avail_actions_dict[e][k])).sample().numpy()
                                        for k in self.agent_keys} for e in range(batch_size)]
                else:
                    explore_actions = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in
                                       range(batch_size)]
            else:
                explore_actions = pi_actions_dict
        elif self.noise_scale is not None:
            if self.use_parameter_sharing:
                key = self.agent_keys[0]
                pi_actions_dict[key] += np.random.normal(0, self.noise_scale, size=pi_actions_dict[key].shape)
            else:
                for key in self.agent_keys:
                    pi_actions_dict[key] += np.random.normal(0, self.noise_scale, size=pi_actions_dict[key].shape)
            explore_actions = pi_actions_dict
        else:
            explore_actions = pi_actions_dict
        return explore_actions

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input,
                                               agent_ids=agents_id,
                                               avail_actions=avail_actions_input,
                                               rnn_hidden=rnn_hidden)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].numpy().reshape([batch_size, self.n_agents])
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].numpy().reshape(batch_size) for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)
        return {"hidden_state": hidden_state, "actions": actions_dict}

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
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    if self.current_step >= self.start_training:
                        train_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(train_info, self.current_step)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
            return

        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict, avail_actions_dict=avail_actions, test_mode=False)
            actions_dict = policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_wandb:
                        step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Results/Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            self._update_explore_factor()

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [0.0 for _ in range(num_envs)], -np.inf
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden = self.init_rnn_hidden(num_envs)

        while episode_count < n_episodes:
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode)
            rnn_hidden, actions_dict = policy_out['hidden_state'], policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                      rewards_dict, terminated_dict, info,
                                      **{'state': state, 'next_state': next_state})
            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_rnn:
                        rnn_hidden = self.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden)
                        if not test_mode:
                            terminal_data = {'obs': next_obs_dict[i],
                                             'episode_step': info[i]['episode_step']}
                            if self.use_global_state:
                                terminal_data['state'] = next_state[i]
                            if self.use_actions_mask:
                                terminal_data['avail_actions'] = next_avail_actions[i]
                            self.memory.finish_path(i, **terminal_data)
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                    else:
                        if self.use_wandb:
                            step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                            step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(step_info, self.current_step)
                        self._update_explore_factor()

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores

    def train_epochs(self, n_epochs=1):
        """
        Train the model for numerous epochs.

        Parameters:
            n_epochs (int): The number of epochs to train.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        for i_epoch in range(n_epochs):
            sample = self.memory.sample()
            if self.use_rnn:
                info_train = self.learner.update_rnn(sample)
            else:
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.e_greedy
        info_train["noise_scale"] = self.noise_scale
        return info_train

    def test(self, env_fn, n_episodes):
        """
        Test the model for some episodes.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes to test.

        Returns:
            scores (List(float)): A list of cumulative rewards for each episode.
        """
        scores = self.run_episodes(env_fn=env_fn, n_episodes=n_episodes, test_mode=True)
        return scores
