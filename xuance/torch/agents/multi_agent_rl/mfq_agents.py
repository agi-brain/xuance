import torch
import numpy as np
from operator import itemgetter
from argparse import Namespace
from tqdm import tqdm
from copy import deepcopy
from gymnasium.spaces import Space
from typing import List, Optional, Dict
from xuance.common import MeanField_OffPolicyBuffer, MeanField_OffPolicyBuffer_RNN, MultiAgentBaseCallback
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, ModuleDict
from xuance.torch.agents import OffPolicyMARLAgents
from xuance.torch.utils import AgentGroupedTensor
from xuance.torch.rl_models import MeanFieldActionValueCritic
from xuance.torch.rl_models.modules import RNN_State, MARLActionOutput
from xuance.torch.rl_models.heads import IndependentMixer
from xuance.torch.rl_models.architectures import MeanFieldQNetwork


class MFQ_Agents(OffPolicyMARLAgents):
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
        super(MFQ_Agents, self).__init__(
            config, envs, num_agents, agent_keys, state_space, observation_space, action_space, callback
        )

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

        self.model = self._build_model()  # build the MARL model
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.agent_grouping, self.model, self.callback)

    def _build_memory(self) -> MeanField_OffPolicyBuffer | MeanField_OffPolicyBuffer_RNN:
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_dict = dict(agent_keys=self.agent_keys,
                          state_space=self.state_space if self.use_global_state else None,
                          obs_space=self.observation_space,
                          act_space=self.action_space,
                          n_envs=self.n_envs,
                          buffer_size=self.buffer_size,
                          batch_size=self.batch_size,
                          avail_actions_shape=avail_actions_shape,
                          use_actions_mask=self.use_actions_mask,
                          max_episode_steps=self.episode_length,
                          n_actions_max=self.n_actions_max)
        Buffer = MeanField_OffPolicyBuffer_RNN if self.use_rnn else MeanField_OffPolicyBuffer
        return Buffer(**input_dict)

    def _build_model(self) -> Module:
        """
        Build the MARL model.

        Returns:
            model (torch.nn.Module): The MARL model.
        """
        q_networks = ModuleDict()
        for group_key, group_agents in self.groups.items():
            reference_agent = group_agents[0]
            # build agent feature encoder as representations
            agent_feature_encoder = self._build_agent_feature_encoder(
                representation_choice=self.config.representation,
                group_agents=group_agents,
                input_space=self.observation_space[reference_agent]
            )
            # build mean actions embedding network
            mean_actions_encoder = self._build_agent_feature_encoder(
                representation_choice="Basic_MLP",
                group_agents=group_agents,
                input_space=(self.n_actions_max,)
            )
            # build inner-group shared q-network
            q_networks[group_key] = MeanFieldActionValueCritic(
                representation=agent_feature_encoder,
                mean_actions_encoder=mean_actions_encoder,
                action_space=self.action_space[reference_agent],
                critic_hidden_size=self.config.q_hidden_size,
                normalizer=self.normalize_fn,
                initializer=self.initializer,
                activation=self.activation,
                device=self.device
            )

        # build MARL model
        model = MeanFieldQNetwork(
            grouping=self.agent_grouping,
            q_networks=q_networks,
            mixer=IndependentMixer(),
            use_rnn=self.use_rnn,
            device=self.device,
            use_distributed_training=self.distributed_training,
            policy_type="Boltzmann",  # "Boltzmann" or "greedy"
            temperature=self.config.temperature,
            n_actions_max=self.n_actions_max
        )

        return model

    def _build_inputs_mean_mask(self,
                                agent_mask: Optional[dict] = None,
                                act_mean_list=None):
        mean_actions_input = {}
        agent_mask_array = torch.as_tensor(np.array([[data[k] for k in self.agent_keys] for data in agent_mask]),
                                           dtype=torch.float, device=self.device)
        # get mean actions as input
        for group in self.group_keys:
            mean_actions_input[group] = torch.as_tensor(np.array([[data[k] for k in self.agent_keys]
                                                                  for data in act_mean_list]), device=self.device)
            if self.use_rnn:
                mean_actions_input[group] = mean_actions_input[group].unsqueeze(2)

        return AgentGroupedTensor(mean_actions_input, self.agent_grouping), agent_mask_array

    def store_experience(self, obs_list, avail_actions, actions_list, obs_next_list,
                         avail_actions_next, rewards_list, terminals_list, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_list (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_list (List[dict]): Actions for each agent in self.agent_keys.
            obs_next_list (List[dict]): Next observations for each agent in self.agent_keys.
            avail_actions_next (List[dict]): The next actions mask values for each agent in self.agent_keys.
            rewards_list (List[dict]): Rewards for each agent in self.agent_keys.
            terminals_list (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_list]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_list]) for k in self.agent_keys},
            'obs_next': {k: np.array([data[k] for data in obs_next_list]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_list]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_list]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data[k] for data in kwargs['agent_mask']]) for k in self.agent_keys},
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
        experience_data['actions_mean'] = {k: np.array([data[k] for data in kwargs['actions_mean']])
                                           for k in self.agent_keys}
        experience_data['actions_mean_next'] = {k: np.array([data[k] for data in kwargs['actions_mean_next']])
                                                for k in self.agent_keys}
        self.memory.store(**experience_data)

    def get_actions(
            self,
                    obs_list: List[dict],
                    agent_mask: Optional[List[dict]] = None,
                    act_mean_list: Optional[List[dict]] = None,
                    avail_actions_list: Optional[List[dict]] = None,
                    rnn_states: Optional[Dict[str, RNN_State]] = None,
                    test_mode: Optional[bool] = False
    ) -> MARLActionOutput:
        batch_size = len(obs_list)
        mean_actions_input, agent_mask_tensor = self._build_inputs_mean_mask(agent_mask, act_mean_list)
        obs_input, agent_indices, avail_actions_input = self._build_inputs(obs_list, avail_actions_list)

        with torch.no_grad():
            model_output = self.model(observations=obs_input,
                                      agent_indices=agent_indices,
                                      mean_actions=mean_actions_input,
                                      avail_actions=avail_actions_input,
                                      rnn_states=rnn_states)
        rnn_states_new = model_output.rnn_states
        actions = model_output.actions

        actions_mean_masked = self.model.get_mean_actions(actions=actions.agent_wise,
                                                          agent_mask_tensor=agent_mask_tensor,
                                                          batch_size=batch_size)

        actions.grouped_tensor = {k: actions.grouped_tensor[k].reshape(batch_size, n).cpu().detach().numpy()
                                  for k, n in self.n_group_agents.items()}
        actions_list = [{k: actions.agent_wise[k][i] for k in self.agent_keys} for i in range(batch_size)]

        actions_mean_masked = {k: v.cpu().detach().numpy() for k, v in actions_mean_masked.items()}
        actions_mean_list = [{k: v[e] for k, v in actions_mean_masked.items()} for e in range(batch_size)]

        if not test_mode:  # get random actions
            actions_list = self.exploration(batch_size, actions_list, avail_actions_list)

        return MARLActionOutput(
            env_actions=actions_list,
            rnn_states=rnn_states_new,
            auxiliary={"actions_mean": actions_mean_list}
        )

    def train(self, train_steps: int) -> dict:
        train_info = {}
        if self.use_rnn:
            with tqdm(total=train_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = train_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(n_episodes=self.n_envs, test_mode=False, close_envs=False)
                    if self.current_step >= self.start_training:
                        update_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(update_info, self.current_step)
                        train_info.update(update_info)
                        self.callback.on_train_epochs_end(self.current_step, model=self.model, memory=self.memory,
                                                          current_episode=self.current_episode, train_steps=train_steps,
                                                          update_info=update_info)

                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(train_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                                n_steps=train_steps, train_info=train_info)
            return train_info

        obs_list = self.train_envs.buf_obs
        agent_mask_list = [data['agent_mask'] for data in self.train_envs.buf_info]
        actions_mean_list = self.actions_mean
        avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None
        state = self.train_envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(train_steps)):
            policy_out = self.get_actions(obs_list=obs_list, agent_mask=agent_mask_list,
                                          act_mean_list=actions_mean_list,
                                          avail_actions_list=avail_actions, test_mode=False)
            actions_list = policy_out.env_actions
            actions_mean_next_list = policy_out.auxiliary["actions_mean"]
            next_obs_list, rewards_list, terminated_list, truncated, info = self.train_envs.step(actions_list)
            next_state = self.train_envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.train_envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.train_envs, model=self.model,
                                        obs=obs_list, next_obs=next_obs_list,
                                        policy_out=policy_out, acts=actions_list, actions_mean_list=actions_mean_list,
                                        rewards=rewards_list, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_list, truncations=truncated, infos=info,
                                        train_steps=train_steps)

            self.store_experience(obs_list, avail_actions, actions_list, next_obs_list, next_avail_actions,
                                  rewards_list, terminated_list, info,
                                  **{'state': state, 'next_state': next_state, 'agent_mask': agent_mask_list,
                                     'actions_mean': actions_mean_list, 'actions_mean_next': actions_mean_next_list})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, model=self.model, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)

            obs_list = deepcopy(next_obs_list)
            agent_mask_list = [data['agent_mask'] for data in info]
            actions_mean_list = deepcopy(actions_mean_next_list)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(self.n_envs):
                if all(terminated_list[i].values()) or truncated[i]:
                    obs_list[i] = info[i]["reset_obs"]
                    self.train_envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.train_envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.train_envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    self.train_envs.buf_info[i]["agent_mask"] = {k: True for k in self.agent_keys}
                    agent_mask_list[i] = {k: True for k in self.agent_keys}
                    actions_mean_list[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        episode_info = {
                            f"Train-Results/Episode-Steps/rank_{self.rank}/env-%d" % i: info[i]["episode_step"],
                            f"Train-Results/Episode-Rewards/rank_{self.rank}/env-%d" % i: info[i]["episode_score"]
                        }
                    else:
                        episode_info = {
                            f"Train-Results/Episode-Steps/rank_{self.rank}": {"env-%d" % i: info[i]["episode_step"]},
                            f"Train-Results/Episode-Rewards/rank_{self.rank}": {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        }
                    self.log_infos(episode_info, self.current_step)
                    train_info.update(episode_info)
                    self.callback.on_train_episode_info(envs=self.train_envs, model=self.model, env_id=i,
                                                        infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        train_steps=train_steps)

            self.current_step += self.n_envs
            self._update_explore_factor()
            self.actions_mean = deepcopy(actions_mean_list)
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, model=self.model,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def run_episodes(self,
                     n_episodes: int = 1,
                     run_envs: Optional[DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv] = None,
                     test_mode: bool = False,
                     close_envs: bool = True) -> list:
        envs = self.train_envs if run_envs is None else run_envs
        num_envs = envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs_list, info = envs.reset()
        agent_mask_list = [data['agent_mask'] for data in info]
        actions_mean_list = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(num_envs)]
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
        rnn_states = self.init_rnn_states(num_envs)

        while current_episode < n_episodes:
            policy_out = self.get_actions(obs_list=obs_list,
                                          agent_mask=agent_mask_list,
                                          act_mean_list=actions_mean_list,
                                          avail_actions_list=avail_actions,
                                          rnn_states=rnn_states,
                                          test_mode=test_mode)
            actions_list = policy_out.env_actions
            actions_mean_next_list = policy_out.auxiliary["actions_mean"]
            rnn_states = policy_out.rnn_states
            next_obs_list, rewards_list, terminated_list, truncated, info = envs.step(actions_list)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_list, avail_actions, actions_list, next_obs_list, next_avail_actions,
                                      rewards_list, terminated_list, info,
                                      **{'state': state, 'next_state': next_state, 'agent_mask': agent_mask_list,
                                         'actions_mean': actions_mean_list,
                                         'actions_mean_next': actions_mean_next_list})

            self.callback.on_test_step(envs=envs, model=self.model, images=images, test_mode=test_mode,
                                       obs=obs_list, policy_out=policy_out, acts=actions_list,
                                       actions_mean_list=actions_mean_list,
                                       next_obs=next_obs_list, rewards=rewards_list,
                                       terminals=terminated_list, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=current_step, current_episode=current_episode)

            obs_list = deepcopy(next_obs_list)
            agent_mask_list = [data['agent_mask'] for data in info]
            actions_mean_list = deepcopy(actions_mean_next_list)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_list[i].values()) or truncated[i]:
                    current_episode += 1
                    obs_list[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.train_envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    agent_mask_list[i] = {k: True for k in self.agent_keys}
                    actions_mean_list[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
                    if self.use_rnn:
                        rnn_states = self.init_rnn_states_item(i_env=i, rnn_states=rnn_states)
                        if not test_mode:
                            terminal_data = {'obs': next_obs_list[i],
                                             'actions_mean': actions_mean_next_list[i],
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
                        self.callback.on_train_episode_info(envs=self.train_envs, model=self.model, env_id=i,
                                                            infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)
            current_step += num_envs

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

            self.callback.on_test_end(envs=envs, model=self.model,
                                      current_train_step=self.current_step,
                                      current_step=current_step, current_episode=current_episode,
                                      scores=scores, best_score=best_score)

            if close_envs:
                envs.close()
        return scores
