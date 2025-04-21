import torch
import numpy as np
import gymnasium as gym
from gymnasium import Space
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from xuance.common import List, Optional, Dict, Union, space2shape, I3CNet_Buffer_RNN
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, REGISTRY_Policy, ModuleDict
from xuance.torch.communications import IC3NetComm
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions
from xuance.torch.agents.base import MARLAgents


class IC3Net_Agents(MARLAgents):

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv]):
        super(IC3Net_Agents, self).__init__(config, envs)
        self.on_policy = True
        self.continuous_control: bool = False
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch
        self.memory = self._build_memory()
        self.policy = self._build_policy()
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

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
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.config.gamma,
                            gae_lam=self.config.gae_lambda,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length,
                            config=self.config,
                            dim_message=self.config.dim_message)
        Buffer = I3CNet_Buffer_RNN
        return Buffer(**input_buffer)

    def _build_communicator(self,
                            input_space: Union[Dict[str, Space], Dict[str, tuple]],
                            config: Namespace
                            ) -> Module:
        communicator = ModuleDict()
        hidden_sizes = {'fc_hidden_sizes': self.config.fc_hidden_sizes,
                        'recurrent_hidden_size': self.config.recurrent_hidden_size}
        for key in self.model_keys:
            input_communicator = dict(
                input_shape=space2shape(input_space[key]),
                hidden_sizes=hidden_sizes,
                comm_passes=self.config.comm_passes,
                device=self.device)
            communicator[key] = IC3NetComm(**input_communicator)
        return communicator

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent

        # build representations
        communicator = self._build_communicator(self.observation_space, self.config)
        space_actor_in = {agent: gym.spaces.Box(-np.inf, np.inf, (self.config.recurrent_hidden_size,), dtype=np.float32) for agent in self.observation_space}
        dim_obs_all = sum([sum(space_actor_in[k].shape) for k in self.agent_keys])
        space_critic_in = {k: (dim_obs_all,) for k in self.agent_keys}
        A_representation = self._build_representation(self.config.representation, space_actor_in, self.config)
        C_representation = self._build_representation(self.config.representation, space_critic_in, self.config)

        # build policies
        if self.config.policy == "IC3Net_Policy":
            policy = REGISTRY_Policy[self.config.policy](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None, communicator=communicator)
            self.continuous_control = False
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, receive_msg, **kwargs):
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'message': receive_msg,
            'log_pi_old': log_pi_a,
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys}
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
        self.memory.store(**experience_data)

    def init_message(self, rnn_hidden_actor):
        message = {k: torch.zeros_like(rnn_hidden_actor[k][0]).detach().cpu().numpy() for k in self.model_keys}
        return message

    def cal_msg(self, message: dict) -> dict:
        receive_msg = {}
        total_message = np.zeros_like(next(iter(message.values())))
        for m in message.values():
            total_message += m
        for k in self.model_keys:
            other_agents_sum = total_message - message[k]
            receive_msg[k] = other_agents_sum / (self.n_agents - 1)
        return receive_msg

    def init_rnn_hidden(self, n_envs):
        rnn_hidden_actor, rnn_hidden_critic = None, None
        if self.use_rnn:
            batch = n_envs * self.n_agents if self.use_parameter_sharing else n_envs
            rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(batch) for k in self.model_keys}
            rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(batch) for k in self.model_keys}
        return rnn_hidden_actor, rnn_hidden_critic

    def init_hidden_item(self,
                         i_env: int,
                         rnn_hidden_actor: Optional[dict] = None,
                         rnn_hidden_critic: Optional[dict] = None):
        assert self.use_rnn is True, "This method cannot be called when self.use_rnn is False."
        if self.use_parameter_sharing:
            b_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
        else:
            b_index = [i_env, ]
        for k in self.model_keys:
            rnn_hidden_actor[k] = self.policy.actor_representation[k].init_hidden_item(b_index, *rnn_hidden_actor[k])
        if rnn_hidden_critic is None:
            return rnn_hidden_actor, None
        for k in self.model_keys:
            rnn_hidden_critic[k] = self.policy.critic_representation[k].init_hidden_item(b_index, *rnn_hidden_critic[k])
        return rnn_hidden_actor, rnn_hidden_critic

    def _build_critic_inputs(self, batch_size: int, obs_batch: dict,
                             state: Optional[np.ndarray]):
        if self.use_global_state:
            if self.use_parameter_sharing:
                key = self.model_keys[0]
                bs = batch_size * self.n_agents
                state_n = np.stack([state for _ in range(self.n_agents)], axis=1).reshape([bs, -1])
                critic_input = {key: state_n}
            else:
                critic_input = {k: state for k in self.model_keys}
        else:
            if self.use_parameter_sharing:
                key = self.model_keys[0]
                bs = batch_size * self.n_agents
                joint_obs = obs_batch[key].reshape([batch_size, self.n_agents, -1]).reshape([batch_size, 1, -1])
                joint_obs = np.repeat(joint_obs, repeats=self.n_agents, axis=1)
            else:
                bs = batch_size
                joint_obs = np.stack(itemgetter(*self.agent_keys)(obs_batch), axis=1)
            joint_obs = joint_obs.reshape([bs, 1, -1]) if self.use_rnn else joint_obs.reshape([bs, -1])
            critic_input = {k: joint_obs for k in self.model_keys}
        return critic_input

    def _build_inputs(self,
                      obs_dict: List[dict],
                      avail_actions_dict: Optional[List[dict]] = None,
                      message: Optional[dict] = None):
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
                obs_input = {key: obs_array.reshape([bs, 1, -1])}
                agents_id = agents_id.reshape(bs, 1, -1)
                if self.use_actions_mask:
                    avail_actions_input = {key: avail_actions_array.reshape([bs, 1, -1])}
            else:
                obs_input = {key: obs_array.reshape([bs, -1])}
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
        message = {k: message[k].transpose(1, 0, 2) for k in self.model_keys}
        return obs_input, agents_id, avail_actions_input, message

    def action(self,
               obs_dict: List[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               receive_message: Optional[dict] = False,
               **kwargs):
        n_env = len(obs_dict)
        rnn_hidden_critic_new, values_out, log_pi_a_dict, values_dict = {}, {}, {}, {}

        obs_input, agents_id, avail_actions_input, message_input = self._build_inputs(obs_dict, avail_actions_dict, message=receive_message)
        # encode obs
        obs_input = self.policy.observation_encode(obs_input)
        obs_input = {k: obs_input[k].detach().cpu().numpy() for k in self.model_keys}
        rnn_hidden_actor_new, pi_dists, message = self.policy(observation=obs_input,
                                                            agent_ids=agents_id,
                                                            avail_actions=avail_actions_input,
                                                            rnn_hidden=rnn_hidden_actor,
                                                            message_input=message_input)
        if not test_mode:
            critic_input = self._build_critic_inputs(batch_size=n_env, obs_batch=obs_input, state=state)
            rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic,
                                                                       message_input=message_input)
            values_dict = {k: values_out[k].cpu().detach().numpy() for k in self.agent_keys}

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
                "actions": actions_dict, "log_pi": log_pi_a_dict, "values": values_dict, "message": message}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None,
                    receive_message: Optional[dict] = False,):
        rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
        obs_input = {k: obs_dict[k][None, :] for k in self.agent_keys} if self.use_rnn else obs_dict
        message_input = {k: receive_message[k].transpose(1, 0, 2) for k in self.model_keys}
        obs_input = self.policy.observation_encode(obs_input)
        obs_input = {k: obs_input[k].detach().cpu().numpy() for k in self.model_keys}
        critic_input = self._build_critic_inputs(batch_size=1, obs_batch=obs_input, state=state)
        rnn_hidden_critic_new, values_out = self.policy.get_values(observation=critic_input,
                                                                   rnn_hidden=rnn_hidden_critic_i,
                                                                   message_input=message_input)
        values_dict = {k: values_out[k].cpu().detach().numpy().reshape([]) for k in self.agent_keys}

        return rnn_hidden_critic_new, values_dict

    def train_epochs(self, n_epochs=1):
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    info_train = self.learner.update_rnn(sample)
            self.memory.clear()
        return info_train

    def train(self, n_steps):
        return_info = {}
        assert self.use_rnn
        with tqdm(total=n_steps) as process_bar:
            step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
            n_steps_all = n_steps * self.n_envs
            while step_last - step_start < n_steps_all:
                self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)
                return_info.update(train_info)
                process_bar.update((self.current_step - step_last) // self.n_envs)
                step_last = deepcopy(self.current_step)
            process_bar.update(n_steps - process_bar.last_print_n)
        return return_info

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
        message_dict = self.init_message(rnn_hidden_actor)
        while episode_count < n_episodes:
            step_info = {}
            receive_msg = self.cal_msg(message_dict)
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     receive_message=receive_msg, test_mode=test_mode)
            rnn_hidden_actor, rnn_hidden_critic = policy_out['rnn_hidden_actor'], policy_out['rnn_hidden_critic']
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            message_dict = policy_out['message']
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
                                      terminated_dict, info, receive_msg, **{'state': state})
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
                            message_dict = self.init_message(rnn_hidden_actor)
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
                                                             rnn_hidden_critic=rnn_hidden_critic,
                                                             receive_message=receive_msg)
                        self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                        if self.use_rnn:
                            rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor,
                                                                                        rnn_hidden_critic)
                            message_dict = self.init_message(rnn_hidden_actor)
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

    def test(self, env_fn, n_episodes):
        scores = self.run_episodes(env_fn=env_fn, n_episodes=n_episodes, test_mode=True)
        return scores
