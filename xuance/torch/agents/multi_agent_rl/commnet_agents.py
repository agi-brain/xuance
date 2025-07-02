from copy import deepcopy
from operator import itemgetter
from typing import Dict, List

import numpy as np
import torch
from gymnasium import Space
from argparse import Namespace

from xuance.torch.agents.multi_agent_rl.ippo_agents import IPPO_Agents

from xuance.common import Optional, Union, space2shape
import gymnasium as gym
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, REGISTRY_Policy, ModuleDict
from xuance.torch.communications.comm_net import CommNet
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions
from xuance.torch.agents.base import BaseCallback


class CommNet_Agents(IPPO_Agents):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):

        super(CommNet_Agents, self).__init__(config, envs, callback)
        self.policy = self._build_policy()
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, callback)

    def _build_communicator(self, input_space: Union[Dict[str, Space], Dict[str, tuple]], ) -> Module:
        communicator = ModuleDict()
        hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                        'recurrent_hidden_size': self.config.recurrent_hidden_size}
        for key in self.model_keys:
            input_communicator = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                model_keys=self.model_keys,
                agent_keys=self.agent_keys,
                n_agents=self.n_agents,
                device=self.device,
                config=self.config)
            communicator[key] = CommNet(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent

        # build representations
        communicator = self._build_communicator(self.observation_space)
        space_actor_in = {agent: gym.spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32)
                          for agent in self.observation_space}
        A_representation = self._build_representation(self.config.representation, space_actor_in, self.config)
        C_representation = self._build_representation(self.config.representation, space_actor_in, self.config)

        # build policies
        if self.config.policy == "CommNet_Policy":
            policy = REGISTRY_Policy[self.config.policy](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                communicator=communicator, agent_keys=self.agent_keys, comm_passes=self.config.comm_passes)
            self.continuous_control = False
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def action(self,
               obs_dict: List[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               info: dict = None,
               **kwargs):
        n_env = len(obs_dict)
        rnn_hidden_critic_new, values_out, log_pi_a_dict, values_dict = {}, {}, {}, {}
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        alive_ally = {k: np.stack([int(data['agent_mask'][k]) for data in info]).reshape([n_env, 1, -1]) for k in
                            self.agent_keys}
        rnn_hidden_actor_new, pi_dists = self.policy(observation=obs_input,
                                                     agent_ids=agents_id,
                                                     avail_actions=avail_actions_input,
                                                     rnn_hidden=rnn_hidden_actor,
                                                     alive_ally=alive_ally)
        if not test_mode:
            critic_input, _, _ = self._build_inputs(obs_dict, avail_actions_dict)
            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic,
                                                                       alive_ally=alive_ally)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_sample = pi_dists[key].stochastic_sample()
            if self.continuous_control:
                actions_out = actions_sample.reshape(n_env, self.n_agents, -1)
            else:
                actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            if not test_mode:
                log_pi_a = pi_dists[key].log_prob(actions_sample).cpu().detach().numpy()
                log_pi_a = log_pi_a.reshape(n_env, self.n_agents)
                log_pi_a_dict = {k: log_pi_a[:, i] for i, k in enumerate(self.agent_keys)}
                values_out[key] = values_out[key].reshape(n_env, self.n_agents)
                values_dict = {k: values_out[key][:, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
        else:
            actions_sample = {k: pi_dists[k].stochastic_sample() for k in self.agent_keys}
            if self.continuous_control:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([-1]) for k in self.agent_keys}
                                for e in range(n_env)]
            else:
                actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([]) for k in self.agent_keys}
                                for e in range(n_env)]
            if not test_mode:
                log_pi_a = {k: pi_dists[k].log_prob(actions_sample[k]).cpu().detach().numpy() for k in self.agent_keys}
                log_pi_a_dict = {k: log_pi_a[k].reshape([n_env]) for i, k in enumerate(self.agent_keys)}
                values_dict = {k: values_out[k].cpu().detach().numpy().reshape([n_env]) for k in self.agent_keys}

        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None,
                    info: Optional[dict] = None):
        n_env = 1
        rnn_hidden_critic_i = None
        alive_ally = {k: np.array([int(info['agent_mask'][k])]).reshape(n_env, 1, -1) for k in self.agent_keys}
        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
            rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                hidden_item_index, *rnn_hidden_critic[key])}
            batch_size = n_env * self.n_agents
            obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
            critic_input = obs_array.reshape([batch_size, 1, -1])
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).reshape(batch_size, 1, -1).to(
                self.device)

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation={key: critic_input},
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic_i,
                                                                       alive_ally=alive_ally)
            values_out = values_out[key].reshape(self.n_agents)
            values_dict = {k: values_out[i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}

        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
                obs_array = np.array([itemgetter(*self.agent_keys)(obs_dict)])
                critic_input = {k: obs_array[:, i:i+1].reshape(n_env, 1, -1) for i, k in enumerate(self.agent_keys)}
            else:
                critic_input_array = np.concatenate([obs_dict[k].reshape(n_env, 1, -1) for k in self.agent_keys],
                                                    axis=1).reshape(n_env, -1)
                critic_input = {k: critic_input_array for k in self.agent_keys}

            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       rnn_hidden=rnn_hidden_critic_i,
                                                                       alive_ally=alive_ally)
            values_dict = {k: values_out[k].cpu().detach().numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [], -np.inf
        obs_dict, info = envs.reset()
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        state = envs.buf_state if self.use_global_state else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden_actor, rnn_hidden_critic = self.init_rnn_hidden(num_envs)
        info = [{'agent_mask': {k: True for k in self.agent_keys}} for _ in range(num_envs)]
        while episode_count < n_episodes:
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     test_mode=test_mode, info=info)
            rnn_hidden_actor, rnn_hidden_critic = policy_out['rnn_hidden_actor'], policy_out['rnn_hidden_critic']
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                      terminated_dict, info, **{'state': state})
            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = deepcopy(next_state) if self.use_global_state else None

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if self.use_rnn:
                            rnn_hidden_actor, _ = self.init_hidden_item(i, rnn_hidden_actor)
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                    else:
                        if all(terminated_dict[i].values()):
                            value_next = {key: 0.0 for key in self.agent_keys}
                        else:
                            _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                             state=None if state is None else state[i],
                                                             rnn_hidden_critic=rnn_hidden_critic, info=info[i])
                        self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                        if self.use_rnn:
                            rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor,
                                                                                        rnn_hidden_critic)
                        if self.use_wandb:
                            step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                            step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(step_info, self.current_step)
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards/Mean-Score": np.mean(scores),
                "Test-Results/Episode-Rewards/Std-Score": np.std(scores),
            }
            self.log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores