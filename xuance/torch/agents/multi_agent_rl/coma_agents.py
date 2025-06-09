import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from torch.nn.functional import one_hot
from xuance.common import List, Optional, Union
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, Tensor
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.utils.distributions import Categorical
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyMARLAgents, BaseCallback


class COMA_Agents(OnPolicyMARLAgents):
    """The implementation of COMA agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(COMA_Agents, self).__init__(config, envs, callback)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.egreedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy

        self.use_global_state = True
        self.continuous_control = False
        self.state_space = envs.state_space

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)
        self.learner.egreedy = self.egreedy

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
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "Categorical_COMA_Policy":
            policy = REGISTRY_Policy["Categorical_COMA_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None,
                dim_global_state=self.state_space.shape[0])
        else:
            raise AttributeError(f"COMA currently does not support the policy named {self.config.policy}.")

        return policy

    def store_experience(self, obs_dict, avail_actions, actions_dict, log_pi_a, rewards_dict, values_dict,
                         terminals_dict, info, **kwargs):
        """
        Store experience data into replay buffer.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions (List[dict]): Actions mask values for each agent in self.agent_keys.
            actions_dict (List[dict]): Actions for each agent in self.agent_keys.
            log_pi_a (dict): The log of pi.
            rewards_dict (List[dict]): Rewards for each agent in self.agent_keys.
            values_dict (dict): Critic values for each agent in self.agent_keys.
            terminals_dict (List[dict]): Terminated values for each agent in self.agent_keys.
            info (List[dict]): Other information for the environment at current step.
            **kwargs: Other inputs.
        """
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            # 'log_pi_old': log_pi_a,
            'rewards': {k: np.array([np.array(list(data.values())).mean() for data in rewards_dict])
                        for k in self.agent_keys},
            'values': values_dict,
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
        self.memory.store(**experience_data)

    def action(self,
               obs_dict: List[dict],
               state: Optional[np.ndarray] = None,
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden_actor: Optional[dict] = None,
               rnn_hidden_critic: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden_actor (Optional[dict]): The RNN hidden states of actor representation.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_actor_new (dict): The new RNN hidden states of actor representation (if self.use_rnn=True).
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            actions_dict (dict): The output actions.
            log_pi_a (dict): The log of pi.
            values_dict (dict): The evaluated critic values (when test_mode is False).
        """
        n_env = len(obs_dict)
        rnn_hidden_critic_new, log_pi_a_dict, values_dict, actions_out = {}, {}, {}, None

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        rnn_hidden_actor_new, pi_probs = self.policy(observation=obs_input,
                                                     agent_ids=agents_id,
                                                     avail_actions=avail_actions_input,
                                                     rnn_hidden=rnn_hidden_actor,
                                                     epsilon=self.egreedy,
                                                     test_mode=test_mode)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            if self.use_actions_mask:
                pi_probs[key][Tensor(avail_actions_input[key]) == 0] = 0.0
            if test_mode:
                actions_sample = pi_probs[key].max(dim=-1)[1]
            else:
                pi_dists = Categorical(probs=pi_probs[key])
                actions_sample = pi_dists.sample()
            actions_out = actions_sample.reshape(n_env, self.n_agents)
            actions_dict = [{k: actions_out[e, i].cpu().detach().numpy() for i, k in enumerate(self.agent_keys)}
                            for e in range(n_env)]
            actions_onehot = {key: one_hot(actions_out, self.action_space[key].n)}
        else:
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(n_env, -1, -1).to(self.device)
            bs = n_env * self.n_agents
            agents_id = agents_id.reshape(bs, 1, -1) if self.use_rnn else agents_id.reshape(bs, -1)
            if self.use_actions_mask:
                for k in self.agent_keys:
                    pi_probs[k][Tensor(avail_actions_input[k]) == 0] = 0.0
            if test_mode:
                actions_sample = {k: pi_probs[k].max(dim=-1)[1] for k in self.agent_keys}
            else:
                pi_dists = {k: Categorical(probs=pi_probs[k]) for k in self.agent_keys}
                actions_sample = {k: pi_dists[k].sample() for k in self.agent_keys}
            actions_out = torch.stack(itemgetter(*self.agent_keys)(actions_sample), dim=-1)
            actions_dict = [{k: actions_sample[k].cpu().detach().numpy()[e].reshape([]) for k in self.agent_keys}
                            for e in range(n_env)]
            actions_onehot = {k: one_hot(actions_sample[k], self.action_space[k].n) for k in self.agent_keys}

        if not test_mode:  # calculate target values
            if self.use_rnn:
                state = Tensor(np.array(state)).reshape(n_env, 1, -1)
                if self.use_parameter_sharing:
                    actions_onehot = {k: actions_onehot[k].unsqueeze(1) for k in self.model_keys}
                else:
                    actions_onehot = {k: actions_onehot[k] for k in self.model_keys}
            else:
                state = Tensor(np.array(state)).reshape(n_env, -1)

            rnn_hidden_critic_new, values_out = self.policy.get_values(state=Tensor(state).to(self.device),
                                                                       observation=obs_input,
                                                                       actions=actions_onehot,
                                                                       agent_ids=agents_id,
                                                                       rnn_hidden=rnn_hidden_critic,
                                                                       target=True)
            if self.use_rnn:
                values_out = values_out.reshape(n_env, self.n_agents, -1)
                actions_out = actions_out.reshape(n_env, self.n_agents)
            values_out = values_out.gather(-1, actions_out.unsqueeze(-1)).reshape(n_env, self.n_agents)
            values_out = values_out.cpu().detach().numpy()
            values_dict = {k: values_out[:, i] for i, k in enumerate(self.agent_keys)}
        return {"rnn_hidden_actor": rnn_hidden_actor_new, "rnn_hidden_critic": rnn_hidden_critic_new,
                "actions": actions_dict, "log_pi": log_pi_a_dict, "values": values_dict}

    def values_next(self,
                    i_env: int,
                    obs_dict: dict,
                    state: Optional[np.ndarray] = None,
                    actions_n: Optional[np.ndarray] = None,
                    rnn_hidden_critic: Optional[dict] = None):
        """
        Returns critic values of one environment that finished an episode.

        Parameters:
            i_env (int): The index of environment.
            obs_dict (dict): Observations for each agent in self.agent_keys.
            state (Optional[np.ndarray]): The global state.
            actions_n (Optional[np.ndarray]): The actions that were token.
            rnn_hidden_critic (Optional[dict]): The RNN hidden states of critic representation.

        Returns:
            rnn_hidden_critic_new (dict): The new RNN hidden states of critic representation (if self.use_rnn=True).
            values_dict: The critic values.
        """
        n_env = 1
        bs = n_env * self.n_agents
        rnn_hidden_critic_i = None

        agents_id = torch.eye(self.n_agents).unsqueeze(0).repeat(n_env, 1, 1).to(self.device)
        if self.use_rnn:
            state = state.reshape(n_env, 1, -1)
            agents_id = agents_id.reshape(bs, 1, -1)
        else:
            state = state.reshape(n_env, -1)
            agents_id.reshape(bs, -1)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(actions_n)))
            if self.use_rnn:
                hidden_item_index = np.arange(i_env * self.n_agents, (i_env + 1) * self.n_agents)
                rnn_hidden_critic_i = {key: self.policy.critic_representation[key].get_hidden_item(
                    hidden_item_index, *rnn_hidden_critic[key])}
                obs_array = np.array(itemgetter(*self.agent_keys)(obs_dict))
                obs_input = {key: obs_array.reshape([bs, 1, -1])}
                actions_tensor = actions_tensor.reshape(n_env, 1, self.n_agents).to(self.device)
            else:
                obs_input = {key: np.array([itemgetter(*self.agent_keys)(obs_dict)])}
                actions_tensor = actions_tensor.reshape(n_env, self.n_agents).to(self.device)
            actions_onehot = {key: one_hot(actions_tensor.long(), self.action_space[key].n)}
        else:
            if self.use_rnn:
                rnn_hidden_critic_i = {k: self.policy.critic_representation[k].get_hidden_item(
                    [i_env, ], *rnn_hidden_critic[k]) for k in self.agent_keys}
                obs_input = {k: obs_dict[k][None, None, :] for k in self.agent_keys}
            else:
                obs_input = {k: obs_dict[k][None, :] for k in self.agent_keys}
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(actions_n))).reshape(n_env, self.n_agents)
            actions_tensor = actions_tensor.to(self.device)
            actions_onehot = {k: one_hot(actions_tensor[:, i].long(), self.action_space[k].n)
                              for i, k in enumerate(self.agent_keys)}

        rnn_hidden_critic_new, values_out = self.policy.get_values(state=Tensor(state).to(self.device),
                                                                   observation=obs_input,
                                                                   actions=actions_onehot,
                                                                   agent_ids=agents_id,
                                                                   rnn_hidden=rnn_hidden_critic_i,
                                                                   target=True)
        if self.use_rnn:
            values_out = values_out.reshape(n_env, self.n_agents, -1)
            actions_tensor = actions_tensor.reshape(n_env, self.n_agents)
        values_out = values_out.gather(-1, actions_tensor.unsqueeze(-1).long())
        values_out = values_out.cpu().detach().numpy().reshape(self.n_agents)
        values_dict = {k: values_out[i] for i, k in enumerate(self.agent_keys)}
        return rnn_hidden_critic_new, values_dict

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
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions, test_mode=False)
            actions_dict, log_pi_a_dict = policy_out['actions'], policy_out['log_pi']
            values_dict = policy_out['values']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None

            self.callback.on_train_step(self.current_step, envs=self.envs, policy=self.policy,
                                        obs=obs_dict, policy_out=policy_out, acts=actions_dict, next_obs=next_obs_dict,
                                        rewards=rewards_dict, state=state, next_state=next_state,
                                        avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                                        terminals=terminated_dict, truncations=truncated, infos=info,
                                        n_steps=n_steps, values_dict=values_dict)

            self.store_experience(obs_dict, avail_actions, actions_dict, log_pi_a_dict, rewards_dict, values_dict,
                                  terminated_dict, info, **{'state': state})
            if self.memory.full:
                for i in range(self.n_envs):
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=next_obs_dict[i],
                                                         state=state_i, actions_n=actions_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
            update_info = self.train_epochs(n_epochs=self.n_epochs)
            self.log_infos(update_info, self.current_step)
            train_info.update(update_info)
            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = self.envs.buf_state if self.use_global_state else None

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    if all(terminated_dict[i].values()):
                        value_next = {key: 0.0 for key in self.agent_keys}
                    else:
                        state_i = state[i] if self.use_global_state else None
                        _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                         state=state_i, actions_n=actions_dict[i])
                    self.memory.finish_path(i_env=i, value_next=value_next,
                                            value_normalizer=self.learner.value_normalizer)
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_global_state:
                        state[i] = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
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
        current_episode, current_step, scores, best_score = 0, 0, [0.0 for _ in range(num_envs)], -np.inf
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

        while current_episode < n_episodes:
            policy_out = self.action(obs_dict=obs_dict, state=state, avail_actions_dict=avail_actions,
                                     rnn_hidden_actor=rnn_hidden_actor, rnn_hidden_critic=rnn_hidden_critic,
                                     test_mode=test_mode)
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

            self.callback.on_test_step(envs=envs, policy=self.policy, images=images, test_mode=test_mode,
                                       obs=obs_dict, policy_out=policy_out, acts=actions_dict,
                                       next_obs=next_obs_dict, rewards=rewards_dict,
                                       terminals=terminated_dict, truncations=truncated, infos=info,
                                       state=state, next_state=next_state,
                                       current_train_step=self.current_step, n_episodes=n_episodes,
                                       current_step=current_step, current_episode=current_episode)

            obs_dict, avail_actions = deepcopy(next_obs_dict), deepcopy(next_avail_actions)
            state = envs.buf_state if self.use_global_state else None

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    current_episode += 1
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if self.use_rnn:
                            rnn_hidden_actor, _ = self.init_hidden_item(i, rnn_hidden_actor)
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, episode_score))
                    else:
                        if all(terminated_dict[i].values()):
                            value_next = {key: 0.0 for key in self.agent_keys}
                        else:
                            _, value_next = self.values_next(i_env=i, obs_dict=obs_dict[i],
                                                             state=state[i], actions_n=actions_dict[i],
                                                             rnn_hidden_critic=rnn_hidden_critic)
                        self.memory.finish_path(i_env=i, i_step=info[i]['episode_step'], value_next=value_next,
                                                value_normalizer=self.learner.value_normalizer)
                        if self.use_rnn:
                            rnn_hidden_actor, rnn_hidden_critic = self.init_hidden_item(i, rnn_hidden_actor,
                                                                                        rnn_hidden_critic)
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
                        self.callback.on_train_episode_info(envs=self.envs, policy=self.policy, env_id=i,
                                                            infos=info, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            n_episodes=n_episodes)
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
            current_step += num_envs

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

            self.callback.on_test_end(envs=envs, policy=self.policy,
                                      current_train_step=self.current_step,
                                      current_step=current_step, current_episode=current_episode,
                                      scores=scores, best_score=best_score)

            if env_fn is not None:
                envs.close()
        return scores

    def train_epochs(self, n_epochs=1):
        """
        Train the model for numerous epochs.

        Returns:
            info_train (dict): The information of training.
        """
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * self.current_step
        info_train = {}
        if self.memory.full:
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_rnn(sample, self.egreedy)
                    else:
                        info_train = self.learner.update(sample, self.egreedy)
            self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                              current_episode=self.current_episode, n_epochs=n_epochs,
                                              buffer_size=self.buffer_size, update_info=info_train)
            self.memory.clear()
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
