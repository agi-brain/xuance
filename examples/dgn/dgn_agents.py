from argparse import Namespace
from copy import deepcopy
from modulefinder import Module
from operator import itemgetter
from typing import Union, Dict, Optional, List

import numpy as np
import torch
from gymnasium import Space, spaces
from torch.nn import ModuleDict

from examples.dgn.dgn_learner import DGN_Learner
from examples.dgn.dgn_policy import DGN_Policy
from xuance.common import space2shape
from xuance.torch import REGISTRY_Policy, REGISTRY_Learners
from xuance.torch.communications.gnn_comm import DGNComm
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions

from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch.agents import IQL_Agents, BaseCallback


class DGN_Agents(IQL_Agents):
    def __init__(self, config: Namespace, envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super().__init__(config, envs, callback)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, callback)


    def _build_communicator(self, input_space: Union[Dict[str, Space], Dict[str, tuple]],) -> ModuleDict:
        communicator = ModuleDict()
        hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                        'recurrent_hidden_size': self.config.recurrent_hidden_size}
        for key in self.model_keys:
            input_communicator = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                atten_head=self.config.attention_head,
                agent_keys=self.agent_keys,
                device=self.device,
                config=self.config)
            communicator[key] = DGNComm(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        space_in = {agent: spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32)
                          for agent in self.observation_space}
        # build representations
        representation = self._build_representation(self.config.representation, space_in, self.config)
        # build communicator
        communicator = self._build_communicator(self.observation_space)

        # build policies
        if self.config.policy == "DGN_Policy":
            REGISTRY_Policy['DGN_Policy'] = DGN_Policy
            policy = REGISTRY_Policy["DGN_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                communicator=communicator, config=self.config, agent_keys=self.agent_keys)
        else:
            raise AttributeError(f"IQL currently does not support the policy named {self.config.policy}.")
        return policy
    
    def _build_learner(self, *args):
        REGISTRY_Learners['DGN_Learner'] = DGN_Learner
        return REGISTRY_Learners[self.config.learner](*args)

    def _build_inputs(self,
                      obs_dict: List[dict],
                      avail_actions_dict: Optional[List[dict]] = None):
        batch_size = len(obs_dict)
        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        avail_actions_input = None

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            obs_array = np.array([itemgetter(*self.agent_keys)(data) for data in obs_dict])
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            avail_actions_array = np.array([itemgetter(*self.agent_keys)(data)
                                            for data in avail_actions_dict]) if self.use_actions_mask else None
            if self.use_rnn:
                obs_input = {key: obs_array.reshape([batch_size, self.n_agents, 1, -1])}
                agents_id = agents_id.reshape(bs, 1, -1)
                if self.use_actions_mask:
                    avail_actions_input = {key: avail_actions_array.reshape([bs, 1, -1])}
            else:
                obs_input = {key: obs_array.reshape([batch_size, self.n_agents, -1])}
                agents_id = agents_id.reshape(bs, -1)
                if self.use_actions_mask:
                    avail_actions_input = {key: avail_actions_array.reshape([bs, -1])}
        else:
            agents_id = None
            if self.use_rnn:
                obs_input = {k: np.stack([data[k] for data in obs_dict]).reshape([bs, 1, -1]) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.stack([data[k] for data in avail_actions_dict]).reshape([bs, 1, -1])
                                           for k in self.agent_keys}
            else:
                obs_input = {k: np.stack([data[k] for data in obs_dict]).reshape(bs, -1) for k in self.agent_keys}
                if self.use_actions_mask:
                    avail_actions_input = {k: np.array([data[k] for data in avail_actions_dict]).reshape([bs, -1])
                                           for k in self.agent_keys}
        return obs_input, agents_id, avail_actions_input

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               info: Optional[dict] = None,
               **kwargs):
        batch_size = len(obs_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        alive_ally = {k: np.stack([int(data['agent_mask'][k]) for data in info]).reshape([batch_size, 1, -1]) for k in
                            self.agent_keys}

        hidden_state, actions, _= self.policy(observation=obs_input,
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

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [], -np.inf
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
        while episode_count < n_episodes:
            step_info = {}
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