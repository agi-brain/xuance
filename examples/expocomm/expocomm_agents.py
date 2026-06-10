import torch
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, MultiAgentBaseCallback, Union
from typing import Dict, List
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv, space2shape
from xuance.torch import Module, ModuleDict
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from expocomm_learner import ExpoComm_Learner
from xuance.torch.policies import QMIX_mixer
from xuance.torch.agents import QMIX_Agents
from expocomm_policy import ExpoCommQnetwork
from expocomm import ExpoComm
import gymnasium as gym
import numpy as np
from operator import itemgetter
from copy import deepcopy


class ExpoComm_Agents(QMIX_Agents):
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
            num_agents: Optional[int] = None,
            agent_keys: Optional[List[str]] = None,
            state_space: Optional[Space] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[MultiAgentBaseCallback] = None
    ):
        super(ExpoComm_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_communicator(self, input_space: Union[Dict[str, Space], Dict[str, tuple]], state_space: Space) -> Module:
        communicator = ModuleDict()
        hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                        'recurrent_hidden_size': self.config.recurrent_hidden_size}
        for key in self.model_keys:
            input_communicator = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                state_shape=self.state_space.shape,
                model_keys=self.model_keys,
                agent_keys=self.agent_keys,
                n_agents=self.n_agents,
                device=self.device,
                config=self.config)
            communicator[key] = ExpoComm(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]

        space_in = {agent: gym.spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32)
                          for agent in self.observation_space}

        # build representations
        representation = self._build_representation(self.config.representation, space_in, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents, self.device)
        
        # Debug: check if q_hidden_size is properly loaded
        if not hasattr(self.config, 'q_hidden_size'):
            raise ValueError("config.q_hidden_size is not defined! Please check your configuration file.")
        if self.config.q_hidden_size is None:
            raise ValueError("config.q_hidden_size is None! Please check your configuration file.")
    
        communicator = self._build_communicator(self.observation_space, self.state_space)
        
        if self.config.policy == "ExpoComm_Q_network":
            policy = ExpoCommQnetwork(
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=self.device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                agent_keys=self.agent_keys, communicator=communicator)
        else:
            raise AttributeError(f"QMIX currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, *args):
        return ExpoComm_Learner(*args)

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               info: Optional[List[dict]] = None,
               **kwargs) -> dict:

        batch_size = len(obs_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        alive_ally = {k: np.stack([int(data['agent_mask'][k]) for data in info]).reshape([batch_size, 1, -1]) for k in
                            self.agent_keys}
        hidden_state, actions, _, _ = self.policy(observation=obs_input,
                                               agent_ids=agents_id,
                                               avail_actions=avail_actions_input,
                                               rnn_hidden=rnn_hidden,
                                               alive_ally=alive_ally)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].reshape([batch_size, self.n_agents]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].reshape(batch_size).cpu().detach().numpy() for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)

        return {"hidden_state": hidden_state, "actions": actions_dict}

    def init_rnn_hidden(self, n_envs) -> Optional[dict]:

        rnn_hidden_states = None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_states = {k: self.policy.representation[k].init_hidden(batch) for k in self.model_keys}

        self.policy.init_msg_prev(rnn_hidden_states)
        return rnn_hidden_states

    def init_hidden_item(self, i_env: int,
                         rnn_hidden: Optional[dict] = None) -> dict:

        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        batch_index = list(range(i_env * self.n_agents, (i_env + 1) * self.n_agents))
        for key in self.model_keys:
            rnn_hidden[key] = self.policy.representation[key].init_hidden_item(batch_index, *rnn_hidden[key])
        key = self.model_keys[0]
        self.policy.communicator[key].msg_prev[batch_index] = 0
        return rnn_hidden

    def run_episodes(self, 
                     n_episodes: int = 1, 
                     run_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                     test_mode: bool = False,
                     close_envs: bool = True) -> list:

        envs = self.train_envs if run_envs is None else run_envs
        num_envs = envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        _current_episode, _current_step, scores, best_score = 0, 0, [], -np.inf
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
        info = [{'agent_mask': {k: True for k in self.agent_keys}} for _ in range(num_envs)]

        while _current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode,
                                     info=info)
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

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=_current_step, current_episode=_current_episode)

            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    _current_episode += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        envs.buf_state[i] = info[i]["reset_state"]
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
                    else:
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                "Train-Results/Episode-Steps/env-%d" % i: info[i]["episode_step"],
                                "Train-Results/Episode-Rewards/env-%d" % i: info[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                "Train-Results/Episode-Steps": {"env-%d" % i: info[i]["episode_step"]},
                                "Train-Results/Episode-Rewards": {
                                    "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                            }
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(episode_info, self.current_step)
                        self._update_explore_factor()
                        self.callback.on_train_episode_info(envs=envs, policy=self.policy, env_id=i,
                                                            infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)
            _current_step += num_envs

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)

            self.callback.on_test_end(envs=envs, policy=self.policy,
                                      current_train_step=self.current_step,
                                      current_step=_current_step, current_episode=_current_episode,
                                      scores=scores, best_score=best_score)

            if close_envs:
                envs.close()
        return scores