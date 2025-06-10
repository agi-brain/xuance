import torch
import numpy as np
from operator import itemgetter
from argparse import Namespace

from sympy.physics.vector import express
from torch.utils.collect_env import env_info_fmt
from tqdm import tqdm
from copy import deepcopy
from xuance.common import List, Union, Optional, MeanField_OffPolicyBuffer
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
        self.actions_mean = {k: np.zeros([self.n_envs, self.n_actions_max]) for k in self.agent_keys}

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
        return MeanField_OffPolicyBuffer(agent_keys=self.agent_keys,
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
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"MFQ currently does not support the policy named {self.config.policy}.")

        return policy

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
            'agent_mask': kwargs['agent_mask'],
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
        experience_data['actions_mean'] = kwargs['actions_mean']
        self.memory.store(**experience_data)

    def action(self,
               obs_dict: List[dict],
               agent_mask: Optional[dict] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               act_mean_dict=None):
        batch_size = len(obs_dict)
        # get mean actions as input
        if self.use_parameter_sharing:
            mean_actions_array = itemgetter(*self.agent_keys)(act_mean_dict)
            key = self.agent_keys[0]
            if self.use_rnn:
                mean_actions_input = {key: np.reshape(mean_actions_array, (batch_size * self.n_agents, 1, -1))}
            else:
                mean_actions_input = {key: np.reshape(mean_actions_array, (batch_size * self.n_agents, -1))}
        else:
            if self.use_rnn:
                mean_actions_input = {k: act_mean_dict[k].reshape([batch_size, 1, -1]) for k in self.agent_keys}
            else:
                mean_actions_input = {k: act_mean_dict[k].reshape([batch_size, -1]) for k in self.agent_keys}
        # get agent masks as tensor
        agent_mask_array = np.array(itemgetter(*self.agent_keys)(agent_mask))
        agent_mask_tensor = torch.Tensor(agent_mask_array).transpose(0, 1).to(self.device)
        agent_mask_repeat = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_actions_max)
        # get other regular inputs
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)

        hidden_state, actions, q_output = self.policy(observation=obs_input,
                                                      agent_ids=agents_id,
                                                      actions_mean=mean_actions_input,
                                                      avail_actions=avail_actions_input,
                                                      rnn_hidden=rnn_hidden)

        # count alive neighbors
        _eyes = torch.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        agent_mask_diagonal = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) * _eyes
        agent_mask_neighbors = agent_mask_tensor.unsqueeze(-1).repeat(1, 1, self.n_agents) - agent_mask_diagonal
        agent_alive_neighbors = agent_mask_neighbors.sum(dim=-1, keepdim=True)
        # calculate mean actions of each agent's neighbors
        actions_sample = self.policy.sample_actions(logits=q_output) * agent_mask_repeat
        actions_sum = actions_sample.sum(dim=-2, keepdim=True).repeat(1, self.n_agents, 1)
        actions_neighbors_sum = actions_sum - actions_sample
        actions_mean_masked = actions_neighbors_sum * agent_mask_repeat / agent_alive_neighbors
        actions_mean_masked = actions_mean_masked.cpu().detach().numpy()
        actions_mean_dict = {k: actions_mean_masked[:, i, :] for i, k in enumerate(self.agent_keys)}

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
        agent_mask_dict = {k: np.array([data['agent_mask'][k] for data in self.envs.buf_info]) for k in self.agent_keys}
        actions_mean_dict = self.actions_mean
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            policy_out = self.action(obs_dict=obs_dict, agent_mask=agent_mask_dict, act_mean_dict=actions_mean_dict,
                                     avail_actions_dict=avail_actions, test_mode=False)
            actions_dict = policy_out['actions']
            actions_mean_dict = policy_out['actions_mean']
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
                                  **{'state': state, 'next_state': next_state,
                                     'actions_mean': actions_mean_dict, 'agent_mask': agent_mask_dict})
            # if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
            #     update_info = self.train_epochs(n_epochs=self.n_epochs)
            #     self.log_infos(update_info, self.current_step)
            #     train_info.update(update_info)
            #     self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
            #                                       current_episode=self.current_episode, n_steps=n_steps,
            #                                       update_info=update_info)

            obs_dict = deepcopy(next_obs_dict)
            self.actions_mean = deepcopy(actions_mean_dict)
            agent_mask_dict = {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys}
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
                    self.envs.buf_info[i]["agent_mask"] = [{k: True for k in self.agent_keys}
                                                           for _ in range(self.n_envs)]
                    #####################
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
            self.callback.on_train_step_end(self.current_step, envs=self.envs, policy=self.policy,
                                            n_steps=n_steps, train_info=train_info)
        return train_info
