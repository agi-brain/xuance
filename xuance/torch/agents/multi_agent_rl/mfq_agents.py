import torch
import numpy as np
from operator import itemgetter
from argparse import Namespace
from tqdm import tqdm
from copy import deepcopy
from xuance.common import List, Union, Optional, MeanField_OffPolicyBuffer, MeanField_OffPolicyBuffer_RNN
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyMARLAgents, BaseCallback


class MFQ_Agents(OffPolicyMARLAgents):
    """The implementation of Mean-Field Q agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(MFQ_Agents, self).__init__(config, envs, callback)

        self.n_actions_list = [a_space.n for a_space in self.action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.actions_mean = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(self.n_envs)]

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_memory(self):
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

    def _build_policy(self) -> Module:
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
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "MF_Q_network":
            policy = REGISTRY_Policy["MF_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                temperature=self.config.temperature,
                policy_type="Boltzmann",  # "Boltzmann" or "greedy"
                action_embedding_hidden_size=self.config.action_embedding_hidden_size)
        else:
            raise AttributeError(f"MFQ currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_inputs_mean_mask(self,
                                agent_mask: Optional[dict] = None,
                                act_mean_dict=None):
        batch_size = len(act_mean_dict)
        agent_mask_array = np.array([itemgetter(*self.agent_keys)(data) for data in agent_mask])
        # get mean actions as input
        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            mean_actions_array = np.array([itemgetter(*self.agent_keys)(data) for data in act_mean_dict])
            if self.use_rnn:
                mean_actions_input = {key: mean_actions_array.reshape([batch_size * self.n_agents, 1, -1])}
            else:
                mean_actions_input = {key: mean_actions_array.reshape([batch_size * self.n_agents, -1])}
        else:
            if self.use_rnn:
                mean_actions_input = {k: np.stack([data[k] for data in act_mean_dict]).reshape([batch_size, 1, -1])
                                      for k in self.agent_keys}
            else:
                mean_actions_input = {k: np.stack([data[k] for data in act_mean_dict]).reshape(batch_size, -1)
                                      for k in self.agent_keys}
        return mean_actions_input, agent_mask_array

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict,
                         avail_actions_next, rewards_dict, terminals_dict, info, **kwargs):
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

    def action(self,
               obs_dict: List[dict],
               agent_mask: Optional[List[dict]] = None,
               act_mean_dict: Optional[List[dict]] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False):
        batch_size = len(obs_dict)
        mean_actions_input, agent_mask_array = self._build_inputs_mean_mask(agent_mask, act_mean_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        agent_mask_tensor = torch.tensor(agent_mask_array, dtype=torch.float32, device=self.device)

        hidden_state, actions, q_output = self.policy(observation=obs_input,
                                                      agent_ids=agents_id,
                                                      actions_mean=mean_actions_input,
                                                      avail_actions=avail_actions_input,
                                                      rnn_hidden=rnn_hidden)
        actions_mean_masked = self.policy.get_mean_actions(actions=actions, agent_mask_tensor=agent_mask_tensor,
                                                           batch_size=batch_size)
        actions_mean_masked = actions_mean_masked.cpu().detach().numpy()
        actions_mean_dict = [{k: actions_mean_masked[e, i] for i, k in enumerate(self.agent_keys)}
                             for e in range(batch_size)]

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].reshape([batch_size, self.n_agents]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].reshape(batch_size).cpu().detach().numpy() for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)

        return {"hidden_state": hidden_state, "actions": actions_dict, "actions_mean": actions_mean_dict}

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        train_info = {}
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    if self.current_step >= self.start_training:
                        update_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(update_info, self.current_step)
                        train_info.update(update_info)
                        self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                          current_episode=self.current_episode, n_steps=n_steps,
                                                          update_info=update_info)

                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
                self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                                n_steps=n_steps, train_info=train_info)
            return train_info

        obs_dict = self.envs.buf_obs
        agent_mask_dict = [data['agent_mask'] for data in self.envs.buf_info]
        actions_mean_dict = self.actions_mean
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            policy_out = self.action(obs_dict=obs_dict, agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                     avail_actions_dict=avail_actions, test_mode=False)
            actions_dict = policy_out['actions']
            actions_mean_next_dict = policy_out['actions_mean']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs_dict, next_obs=next_obs_dict,
                                        policy_out=policy_out, acts=actions_dict, actions_mean_dict=actions_mean_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        n_steps=n_steps)

            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state, 'agent_mask': agent_mask_dict,
                                     'actions_mean': actions_mean_dict, 'actions_mean_next': actions_mean_next_dict})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, n_steps=n_steps,
                                                  update_info=update_info)

            obs_dict = deepcopy(next_obs_dict)
            agent_mask_dict = [data['agent_mask'] for data in info]
            actions_mean_dict = deepcopy(actions_mean_next_dict)
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
                    self.envs.buf_info[i]["agent_mask"] = {k: True for k in self.agent_keys}
                    agent_mask_dict[i] = {k: True for k in self.agent_keys}
                    actions_mean_dict[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
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
                    self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                        infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                        current_step=self.current_step,
                                                        current_episode=self.current_episode,
                                                        n_steps=n_steps)

            self.current_step += self.n_envs
            self._update_explore_factor()
            self.actions_mean = deepcopy(actions_mean_dict)
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            n_steps=n_steps, train_info=train_info)
        return train_info

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
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs_dict, info = envs.reset()
        agent_mask_dict = [data['agent_mask'] for data in info]
        actions_mean_dict = [{k: np.zeros(self.n_actions_max) for k in self.agent_keys} for _ in range(num_envs)]
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

        while current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict,
                                     agent_mask=agent_mask_dict,
                                     act_mean_dict=actions_mean_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode)
            rnn_hidden, actions_dict = policy_out['hidden_state'], policy_out['actions']
            actions_mean_next_dict = policy_out['actions_mean']
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
                                      **{'state': state, 'next_state': next_state, 'agent_mask': agent_mask_dict,
                                         'actions_mean': actions_mean_dict,
                                         'actions_mean_next': actions_mean_next_dict})

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       actions_mean_dict=actions_mean_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=current_step, current_episode=current_episode)

            obs_dict = deepcopy(next_obs_dict)
            agent_mask_dict = [data['agent_mask'] for data in info]
            actions_mean_dict = deepcopy(actions_mean_next_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    current_episode += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    agent_mask_dict[i] = {k: True for k in self.agent_keys}
                    actions_mean_dict[i] = {k: np.zeros(self.n_actions_max) for k in self.agent_keys}
                    if self.use_rnn:
                        rnn_hidden = self.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden)
                        if not test_mode:
                            terminal_data = {'obs': next_obs_dict[i],
                                             'actions_mean': actions_mean_next_dict[i],
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
                            print("Episode: %d, Score: %.2f" % (current_episode, episode_score))
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
                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
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

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)

            self.callback.on_test_end(envs=envs, policy=self.policy,
                                      current_train_step=self.current_step,
                                      current_step=current_step, current_episode=current_episode,
                                      scores=scores, best_score=best_score)

            if env_fn is not None:
                envs.close()
        return scores
