import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import Optional, List
from torch.distributions import Categorical
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Tensor
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import IQL_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN


class IQL_Agents(MARLAgents):
    """The implementation of Independent Q-Learning agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(IQL_Agents, self).__init__(config, envs)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()
        optimizer, scheduler = {}, {}
        for key in self.model_keys:
            optimizer[key] = torch.optim.Adam(self.policy.parameters_model[key], config.learning_rate, eps=1e-5)
            scheduler[key] = torch.optim.lr_scheduler.LinearLR(optimizer[key], start_factor=1.0, end_factor=0.5,
                                                               total_iters=self.config.running_steps)

        # create experience replay buffer
        input_buffer = dict(agent_keys=self.agent_keys,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.config.buffer_size,
                            batch_size=self.config.batch_size,
                            n_actions={k: self.action_space[k].n for k in self.agent_keys},
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=envs.max_episode_steps)
        buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        self.memory = buffer(**input_buffer)

        # initialize the hidden states of the RNN is use RNN-based representations.
        self.rnn_hidden_state = self.init_rnn_hidden(self.n_envs)

        # create learner
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
        representation = self._build_representation(self.config.representation, self.config)

        # build policies
        if self.config.policy == "Basic_Q_network_marl":
            policy = REGISTRY_Policy["Basic_Q_network_marl"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"IQL currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return IQL_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict, avail_actions_next,
                         rewards_dict, terminals_dict, info):
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
            'obs': {k: np.array([itemgetter(k)(data) for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([itemgetter(k)(data) for data in actions_dict]) for k in self.agent_keys},
            'obs_next': {k: np.array([itemgetter(k)(data) for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([itemgetter(k)(data) for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([itemgetter(k)(data) for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([itemgetter(k)(data['agent_mask']) for data in info])
                           for k in self.agent_keys},
            'avail_actions': {k: np.array([itemgetter(k)(data) for data in avail_actions]) for k in self.agent_keys},
            'avail_actions_next': {k: np.array([itemgetter(k)(data) for data in avail_actions_next])
                                   for k in self.agent_keys}
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        self.memory.store(**experience_data)

    def init_rnn_hidden(self, n_envs):
        """
        Returns initialized hidden states of RNN if use RNN-based representations.

        Parameters:
            n_envs (int): The number of parallel environments.
        """
        rnn_hidden_states = {}
        for key in self.model_keys:
            if self.use_rnn:
                batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
                rnn_hidden_states[key] = self.policy.representation[key].init_hidden(batch)
            else:
                rnn_hidden_states[key] = [None, None]
        return rnn_hidden_states

    def init_hidden_item(self, i_env):
        """
        Returns initialized hidden states of RNN for i-th environment.

        Parameters:
            i_env (int): The index of environment that to be selected.
        """
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        for key in self.model_keys:
            self.rnn_hidden_state[key] = self.policy.representation[key].init_hidden_item(
                i_env, *self.rnn_hidden_state[key])

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
        n_env = len(obs_dict)
        avail_actions_input = None

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_rnn:
                batch_size = n_env * self.n_agents
                obs_array = np.array([itemgetter(*self.agent_keys)(data) for data in obs_dict])
                obs_input = {key: obs_array.reshape([batch_size, 1, -1])}
                if self.use_actions_mask:
                    avail_actions_array = np.array([itemgetter(*self.agent_keys)(data) for data in avail_actions_dict])
                    avail_actions_input = {key: avail_actions_array.reshape([batch_size, 1, -1])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(self.device)
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])}
                if self.use_actions_mask:
                    avail_actions_input = {key: np.array([itemgetter(*self.agent_keys)(data) for data in avail_actions_dict])}
                agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            hidden_state, actions, _ = self.policy(observation=obs_input,
                                                   agent_ids=agents_id,
                                                   avail_actions=avail_actions_input,
                                                   rnn_hidden=rnn_hidden)

            actions_out = actions[key].reshape([n_env, self.n_agents]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]
        else:
            if self.use_rnn:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict])[:, None] for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict])[:, None] for k in self.agent_keys}
            else:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict]) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict]) for k in self.agent_keys}

            hidden_state, actions, _ = self.policy(observation=obs_input,
                                                   avail_actions=avail_actions_input,
                                                   rnn_hidden=rnn_hidden)

            actions_out = {}
            for key in self.agent_keys:
                if self.use_rnn:
                    actions_out[key] = actions[key].squeeze(1).cpu().detach().numpy()
                else:
                    actions_out[key] = actions[key].cpu().detach().numpy()
            actions_dict = [{k: actions_out[k][e] for k in self.agent_keys} for e in range(n_env)]

        if not test_mode:  # get random actions
            if np.random.rand() < self.egreedy:
                if self.use_actions_mask:
                    actions_dict = [{k: Categorical(Tensor(avail_actions_dict[e][k])).sample().numpy() for k in self.agent_keys} for e in range(n_env)]
                else:
                    actions_dict = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(n_env)]
        return hidden_state, actions_dict

    def train_epochs(self, n_epoch=1):
        """
        Train the model for numerous epochs.

        Parameters:
            n_epoch (int): The number of epochs to train.

        Returns:
            info_train (dict): The information of training.
        """
        info_train = {}
        for i_epoch in range(n_epoch):
            sample = self.memory.sample()
            if self.use_rnn:
                info_train = self.learner.update_rnn(sample)
            else:
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions
        for _ in tqdm(range(n_steps)):
            step_info = {}
            rnn_hidden_next, actions_dict = self.action(obs_dict=obs_dict,
                                                        avail_actions_dict=avail_actions,
                                                        rnn_hidden=self.rnn_hidden_state,
                                                        test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_avail_actions = self.envs.buf_avail_actions
            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info)
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epoch=1)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)
            avail_actions = deepcopy(next_avail_actions)
            self.rnn_hidden_state = deepcopy(rnn_hidden_next)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if self.use_rnn:
                        terminal_data = {
                            'obs': next_obs_dict[i],
                            'avail_actions': next_avail_actions[i],
                            'episode_step': info[i]['episode_step']
                        }
                        self.memory.finish_path(i, **terminal_data)
                        self.init_hidden_item(i)
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
            if self.egreedy >= self.end_greedy:
                self.egreedy -= self.delta_egreedy

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
        rnn_hidden_state = self.init_rnn_hidden(num_envs)
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < n_episodes:
            rnn_hidden_next, actions_dict = self.action(obs_dict=obs_dict,
                                                        avail_actions_dict=avail_actions,
                                                        rnn_hidden=rnn_hidden_state,
                                                        test_mode=True)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = test_envs.step(actions_dict)
            next_avail_actions = test_envs.buf_avail_actions
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs_dict = deepcopy(next_obs_dict)
            avail_actions = deepcopy(next_avail_actions)
            rnn_hidden_state = deepcopy(rnn_hidden_next)
            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if self.use_rnn:
                        for key in self.model_keys:
                            rnn_hidden_state[key] = self.policy.representation[key].init_hidden_item(
                                i, *rnn_hidden_state[key])
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
