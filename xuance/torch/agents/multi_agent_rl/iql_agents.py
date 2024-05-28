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
from xuance.torch.learners import IQL_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN


class IQL_Agents(MARLAgents):
    """The implementation of Independent Q-Networks agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        super(IQL_Agents, self).__init__(config, envs)
        self.use_actions_mask = config.use_actions_mask

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        # build policy, optimizers, schedulers
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
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
                            max_episode_length=envs.max_episode_length)
        buffer = MARL_OffPolicyBuffer_RNN if self.use_rnn else MARL_OffPolicyBuffer
        self.memory = buffer(**input_buffer)

        # initialize the hidden states of the RNN is use RNN-based representations.
        self.rnn_hidden_state = self.init_rnn_hidden(self.n_envs)

        # create learner
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
        representation = {key: None for key in self.model_keys}
        for key in self.model_keys:
            input_shape = self.observation_space[key].shape
            if self.config.representation == "Basic_Identical":
                representation[key] = REGISTRY_Representation["Basic_Identical"](input_shape=input_shape,
                                                                                 device=self.device)
            elif self.config.representation == "Basic_MLP":
                representation[key] = REGISTRY_Representation["Basic_MLP"](
                    input_shape=input_shape, hidden_sizes=self.config.representation_hidden_size,
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
            elif self.config.representation == "Basic_RNN":
                representation[key] = REGISTRY_Representation["Basic_RNN"](
                    input_shape=input_shape,
                    hidden_sizes={'fc_hidden_sizes': self.config.fc_hidden_sizes,
                                  'recurrent_hidden_size': self.config.recurrent_hidden_size},
                    normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                    N_recurrent_layers=self.config.N_recurrent_layers,
                    dropout=self.config.dropout, rnn=self.config.rnn)
            else:
                raise f"The IQL currently does not support the representation of {self.config.representation}."

        # build policies
        if self.config.policy == "Basic_Q_network_marl":
            policy = REGISTRY_Policy["Basic_Q_network_marl"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise f"The IQL currently does not support the policy named {self.config.policy}."

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
            self.rnn_hidden_state[key] = self.policy.representation[key].init_hidden_item(i_env,
                                                                                          *self.rnn_hidden_state[key])

    def action(self, obs_dict,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        n_env = len(obs_dict)

        if self.use_parameter_sharing:
            # prepare input tensors
            key = self.agent_keys[0]
            if self.use_rnn:
                obs_array = np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])
                raise NotImplementedError
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])}
            if self.use_actions_mask:
                avail_actions_input = {
                    key: np.array([itemgetter(*self.agent_keys)(mask) for mask in avail_actions_dict])}
            else:
                avail_actions_input = None
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)

            hidden_state, actions, _ = self.policy(obs_input, agents_id,
                                                   avail_actions=avail_actions_input,
                                                   rnn_hidden=rnn_hidden)
            if self.use_rnn:
                raise NotImplementedError
            else:
                actions_array = actions[self.agent_keys[0]].cpu().detach().numpy()
                actions_dict = [{k: actions_array[e, i] for i, k in enumerate(self.agent_keys)} for e in range(n_env)]
        else:
            # prepare input tensors
            if self.use_rnn:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict])[:, None]
                             for k in self.agent_keys}
            else:
                obs_input = {k: np.array([itemgetter(k)(env_obs) for env_obs in obs_dict]) for k in self.agent_keys}
            if self.use_actions_mask:
                if self.use_rnn:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict])[:, None]
                                           for k in self.agent_keys}
                else:
                    avail_actions_input = {k: np.array([itemgetter(k)(mask) for mask in avail_actions_dict])
                                           for k in self.agent_keys}
            else:
                avail_actions_input = None
            hidden_state, actions, _ = self.policy(obs_input,
                                                   avail_actions=avail_actions_input,
                                                   rnn_hidden=rnn_hidden)
            actions_dict_ = {}
            for key in self.model_keys:
                if self.use_rnn:
                    actions_dict_[key] = actions[key].squeeze(1).cpu().detach().numpy()
                else:
                    actions_dict_[key] = actions[key].cpu().detach().numpy()
            actions_dict = [{k: actions_dict_[k][e] for k in self.agent_keys} for e in range(n_env)]

        if not test_mode:  # get random actions
            if np.random.rand() < self.egreedy:
                if self.use_actions_mask:
                    actions_random = [{k: Categorical(Tensor(avail_actions_dict[e][k])).sample().numpy()
                                       for k in self.agent_keys} for e in range(n_env)]
                else:
                    actions_random = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(n_env)]
                return hidden_state, actions_random
            else:
                return hidden_state, actions_dict
        else:
            return hidden_state, actions_dict

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
                sample = self.memory.sample()
                if self.use_rnn:
                    step_info = self.learner.update_rnn(sample)
                else:
                    step_info = self.learner.update(sample)
                step_info["epsilon_greedy"] = self.egreedy
                self.log_infos(step_info, self.current_step)
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
                self.egreedy = self.egreedy - self.delta_egreedy

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
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_avail_actions = self.envs.buf_avail_actions
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
                        for key in self.agent_keys:
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
