import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from typing import List
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy, QMIX_mixer
from xuance.torch.learners import QMIX_Learner
from xuance.torch.agents import MARLAgents
from xuance.torch.agents.multi_agent_rl.iql_agents import IQL_Agents
from xuance.common import MARL_OffPolicyBuffer, MARL_OffPolicyBuffer_RNN


class QMIX_Agents(IQL_Agents, MARLAgents):
    """The implementation of QMIX agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        MARLAgents.__init__(self, config, envs)
        self.state_space = envs.state_space

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()
        optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=self.config.running_steps)

        # create experience replay buffer
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space,
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
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents, device)
        if self.config.policy == "Mixing_Q_network":
            policy = REGISTRY_Policy["Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"QMIX currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_learner(self, config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler):
        return QMIX_Learner(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)

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
            **kwargs: Other inputs.
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
                                   for k in self.agent_keys},
            'state': np.array(kwargs['state']),
            'state_next': np.array(kwargs['next_state']),
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        self.memory.store(**experience_data)

    def train(self, n_steps):
        """
        Train the model for numerous steps.

        Parameters:
            n_steps (int): The number of steps to train the model.
        """
        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions
        state = self.envs.buf_state
        for _ in tqdm(range(n_steps)):
            step_info = {}
            rnn_hidden_next, actions_dict = self.action(obs_dict=obs_dict,
                                                        avail_actions_dict=avail_actions,
                                                        rnn_hidden=self.rnn_hidden_state,
                                                        test_mode=False)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state
            next_avail_actions = self.envs.buf_avail_actions
            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epoch=1)
                self.log_infos(train_info, self.current_step)
            obs_dict = deepcopy(next_obs_dict)
            avail_actions = deepcopy(next_avail_actions)
            state = deepcopy(next_state)
            self.rnn_hidden_state = deepcopy(rnn_hidden_next)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if self.use_rnn:
                        terminal_data = {
                            'state': next_state[i],
                            'obs': next_obs_dict[i],
                            'avail_actions': next_avail_actions[i],
                            'episode_step': info[i]['episode_step']
                        }
                        self.memory.finish_path(i, **terminal_data)
                        self.init_hidden_item(i)
                    obs_dict[i] = info[i]["reset_obs"]
                    avail_actions[i] = info[i]["reset_avail_actions"]
                    state = info[i]["reset_state"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    self.envs.buf_state[i] = info[i]["reset_state"]
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
