MAgent2
=================================================

.. image:: ../../../figures/magent/battle.gif
    :height: 150px
.. image:: ../../../figures/magent/battlefield.gif
    :height: 150px
.. image:: ../../../figures/magent/tiger_deer.gif
    :height: 150px

.. raw:: html

    <br><hr>

magent_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.magent2.magent_env.MAgent_Env(env_id, seed, kwargs)

    Interface with the MAgent library,
    which provides a platform for developing and testing multi-agent reinforcement learning algorithms in various scenarios.

    :param env_id: environment id.
    :type env_id: str
    :param seed: use to control randomness within the environment.
    :type seed: int
    :param kwargs: a variable-length keyword argument.
    :type kwargs: dict

.. py:function::
    xuance.environment.magent2.magent_env.MAgent_Env.reset(seed=None, options=None)

    Reset the environment to its initial state.

    :param seed: specifies the seed for the environment's random number generator.
    :type seed: int
    :param options: pass additional options  to the reset process.
    :type options: dict
    :return: the reset local observations and information.
    :rtype: tuple

.. py:function::
    xuance.environment.magent2.magent_env.MAgent_Env.step(actions)

    Take a dictionary of actions as input and performs a step in the environment.

    :param actions: the executable actions for the environment.
    :type actions: np.ndarray
    :return: the next step data, including local observations, global state, rewards, terminated variables, truncated variables, and the other information.
    :rtype: tuple

.. raw:: html

    <br><hr>

magent_vec_env.py
-------------------------------------------------

.. py:class::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent(env_fns)

    A custom environment class that extends the functionality of the DummyVecEnv_Pettingzoo class and is designed to work with the MAgent library.

    :param env_fns: environment function.

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.empty_dict_buffers(i_env)

    Reset the buffers for dictionary data.

    :param i_env: the index of a environment.
    :type i_env: int

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.reset()

    Reset the vectorized environments.

    :return: the reset observations, global states, and the information.
    :rtype: tuple

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.reset_one_env(e)

    Reset a specific environment within the vectorized environment.

    :param e:  Index of the specific environment within the vectorized environment.
    :type e: int
    :return: a list containing observations for each agent in the specified environment.
    :rtype: list

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent._get_max_obs_shape(k, observation_shape)

    Determine the maximum shape of observations among a set of agents in the environment.

    :param k: a list of keys corresponding to agents.
    :type k: list
    :param observation_shape: the shape of observations for all agents.
    :type observation_shape: tuple
    :return: the maximum shape among the observations of the specified agents.
    :rtype: int

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.step_async(actions)

    Sends asynchronous step commands to each subprocess with the specified actions.

    :param actions: the executable actions for n parallel environments.
    :type actions: np.ndarray

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.step_wait()

    Waits for the completion of asynchronous step operations and updates internal buffers with the received results.

    :return: the observations, states, rewards, terminal flags, truncation flags, and information.
    :rtype: tuple

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.render(mode)

    Generate visual representations of the environment.

    :param mode: an optional argument that specifies the rendering mode.
    :type mode: str
    :return: a list of rendered outputs for each environment.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.global_state()

    Return the global state of the parallel environments.

    :return: the global state of the parallel environments.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.global_state_one_env(e)

    Return the global state of the parallel environments.

    :param e: the index of the environment for which you want to retrieve the global state.
    :type e: int
    :return: the global state of the specified environment converted to a numpy array.
    :rtype: np.ndarray

.. py:function::
    xuance.environment.magent2.magent_vec_env.DummyVecEnv_MAgent.agent_mask()

    Return the agent mask.

    :return: the agent mask.
    :rtype: np.ndarray

.. raw:: html

    <br><hr>

Source Code
---------------------------------------------

.. tabs::

    .. group-tab:: magent_env.py

        .. code-block:: python

            from pettingzoo.utils.env import ParallelEnv
            from xuance.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
            from xuance.environment.magent2 import AGENT_NAME_DICT
            import importlib


            class MAgent_Env(PettingZoo_Env, ParallelEnv):
                metadata = {"render_modes": ["human"], "name": "rps_v2"}
                def __init__(self, env_id: str, seed: int, **kwargs):
                    scenario = importlib.import_module('xuance.environment.magent2.environments.' + env_id)

                    if env_id in ["adversarial_pursuit_v4"]:
                        kwargs['minimap_mode'] = False
                        kwargs['tag_penalty'] = -0.2
                    if env_id in ["battle_v4", "battlefield_v4", "combined_arms_v6"]:
                        kwargs['step_reward'] = -0.005
                        kwargs['dead_penalty'] = -0.1
                        kwargs['attack_peanlty'] = -0.1
                        kwargs['attack_opponent_reward'] = 0.2
                    if env_id in ["gather_v4"]:
                        kwargs['step_reward'] = -0.01
                        kwargs['dead_penalty'] = -1
                        kwargs['attack_peanlty'] = -0.1
                        kwargs['attack_food_reward'] = 0.5
                    if env_id in ["tiger_deer_v3"]:
                        kwargs['tiger_step_recover'] = -0.1
                        kwargs['deer_attacked'] = -0.1

                    self.env = scenario.env(**kwargs).unwrapped
                    self.scenario_name = 'magent2.' + env_id
                    self.n_handles = len(self.env.handles)
                    self.side_names = AGENT_NAME_DICT[env_id]
                    self.env.reset(seed)

                    self.state_space = self.env.state_space
                    self.action_spaces = {k: self.env.action_spaces[k] for k in self.env.agents}
                    self.observation_spaces = {k: self.env.observation_spaces[k] for k in self.env.agents}
                    self.agents = self.env.agents
                    self.n_agents_all = len(self.agents)

                    self.handles = self.env.handles

                    self.agent_ids = [self.env.env.get_agent_id(h) for h in self.handles]
                    self.n_agents = [self.env.env.get_num(h) for h in self.handles]

                    self.metadata = self.env.metadata
                    self.max_cycles = self.env.max_cycles
                    self.individual_episode_reward = {k: 0.0 for k in self.agents}

                def reset(self, seed=None, option=None):
                    observations = self.env.reset(seed, option)
                    for agent_key in self.agents:
                        self.individual_episode_reward[agent_key] = 0.0
                        observations[agent_key] = observations[agent_key].reshape([-1])
                    reset_info = {
                        "infos": {},
                        "individual_episode_rewards": self.individual_episode_reward
                    }
                    return observations, reset_info

                def step(self, actions):
                    observations, rewards, terminations, truncations, infos = self.env.step(actions)
                    for k, v in rewards.items():
                        self.individual_episode_reward[k] += v
                        observations[k] = observations[k].reshape([-1])
                    step_info = {"infos": infos,
                                "individual_episode_rewards": self.individual_episode_reward}
                    return observations, rewards, terminations, truncations, step_info



    .. group-tab:: magent_vec_env.py

        .. code-block:: python

            import copy

            from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
            from xuance.environment.vector_envs.env_utils import obs_n_space_info
            from xuance.environment.pettingzoo.pettingzoo_vec_env import DummyVecEnv_Pettingzoo
            from operator import itemgetter
            import numpy as np
            import time


            class DummyVecEnv_MAgent(DummyVecEnv_Pettingzoo):
                def __init__(self, env_fns):
                    self.waiting = False
                    self.envs = [fn() for fn in env_fns]
                    env = self.envs[0]
                    self.handles = env.handles
                    VecEnv.__init__(self, len(env_fns), env.observation_spaces, env.action_spaces)
                    self.state_space = env.state_space
                    self.state_shape = self.state_space.shape
                    self.state_dtype = self.state_space.dtype
                    obs_n_space = env.observation_spaces  # [Box(dim_o), Box(dim_o), ...] ----> dict
                    self.agent_ids = env.agent_ids
                    self.n_agents = [env.get_num(h) for h in self.handles]
                    self.side_names = env.side_names

                    self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)
                    self.agent_keys = [[self.keys[k] for k in ids] for ids in self.agent_ids]
                    self.n_agent_all = len(self.keys)
                    # max_obs_shape = self._get_max_obs_shape(self.keys, self.observation_space)
                    self.obs_shapes = [self.shapes[self.agent_keys[h.value][0]] for h in self.handles]
                    self.obs_dtype = self.dtypes[self.keys[0]]

                    # buffer of dict data
                    self.buf_obs_dict = [{k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys} for _ in
                                        range(self.num_envs)]
                    self.buf_rews_dict = [{k: 0.0 for k in self.keys} for _ in range(self.num_envs)]
                    self.buf_dones_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
                    self.buf_trunctions_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
                    self.buf_infos_dict = [{} for _ in range(self.num_envs)]
                    # buffer of numpy data
                    self.buf_state = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
                    self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool) for n in self.n_agents]
                    self.buf_obs = [np.zeros((self.num_envs, n, np.prod(self.obs_shapes[h])), dtype=self.obs_dtype) for h, n in
                                    enumerate(self.n_agents)]
                    self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in self.n_agents]
                    self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]
                    self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool) for n in self.n_agents]

                    self.max_episode_length = env.max_cycles
                    self.actions = None

                def empty_dict_buffers(self, i_env):
                    # buffer of dict data
                    self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
                    self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
                    self.buf_dones_dict[i_env] = {k: False for k in self.keys}
                    self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
                    self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

                def reset(self):
                    for e in range(self.num_envs):
                        obs, info = self.envs[e].reset()
                        self.buf_obs_dict[e].update(obs)
                        self.buf_infos_dict[e].update(info["infos"])
                        for h, agent_keys_h in enumerate(self.agent_keys):
                            self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
                    return self.buf_obs.copy(), self.buf_infos_dict.copy()

                def reset_one_env(self, e):
                    o = self.envs[e].reset()
                    self.buf_obs_dict[e].update(o)
                    obs_e = []
                    for h, agent_keys_h in enumerate(self.agent_keys):
                        self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
                        obs_e.append(self.buf_obs[h][e])

                    return obs_e

                def _get_max_obs_shape(self, k, observation_shape):
                    obs_shape_n = itemgetter(*list(k))(observation_shape)
                    size_obs_n = []
                    for shape in obs_shape_n:
                        size_obs_n.append(shape.shape)
                    return max(size_obs_n)

                def step_async(self, actions):
                    if self.waiting:
                        raise AlreadySteppingError
                    listify = True
                    try:
                        if len(actions) == self.num_envs:
                            listify = False
                    except TypeError:
                        pass
                    if not listify:
                        self.actions = actions
                    else:
                        assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                            actions, self.num_envs)
                        self.actions = [actions]
                    self.waiting = True

                def step_wait(self):
                    if not self.waiting:
                        raise NotSteppingError

                    for e in range(self.num_envs):
                        action_n = self.actions[e]
                        o, r, d, t, info = self.envs[e].step(action_n)
                        if len(o.keys()) < self.n_agent_all:
                            self.empty_dict_buffers(e)
                        # update the data of alive agents
                        self.buf_obs_dict[e].update(o)
                        self.buf_rews_dict[e].update(r)
                        self.buf_dones_dict[e].update(d)
                        self.buf_trunctions_dict[e].update(t)
                        self.buf_infos_dict[e].update(info["infos"])

                        # resort the data as group-wise
                        episode_scores = []
                        mask = self.envs[e].get_agent_mask()
                        for h, agent_keys_h in enumerate(self.agent_keys):
                            getter = itemgetter(*agent_keys_h)
                            self.buf_agent_mask[h][e] = mask[self.agent_ids[h]]
                            self.buf_obs[h][e] = getter(self.buf_obs_dict[e])
                            self.buf_rews[h][e, :, 0] = getter(self.buf_rews_dict[e])
                            self.buf_dones[h][e] = getter(self.buf_dones_dict[e])
                            self.buf_trunctions[h][e] = getter(self.buf_trunctions_dict[e])
                            episode_scores.append(getter(info["individual_episode_rewards"]))
                        self.buf_infos_dict[e]["individual_episode_rewards"] = episode_scores

                        if all(self.buf_dones_dict[e].values()) or all(self.buf_trunctions_dict[e].values()):
                            obs_reset, _ = self.envs[e].reset()
                            state_reset = self.envs[e].state()
                            mask_reset = self.envs[e].get_agent_mask()
                            obs_reset_handles, mask_reset_handles = [], []
                            for h, agent_keys_h in enumerate(self.agent_keys):
                                getter = itemgetter(*agent_keys_h)
                                obs_reset_handles.append(np.array(getter(obs_reset)))
                                mask_reset_handles.append(mask_reset[self.agent_ids[h]])

                            self.buf_infos_dict[e]["reset_obs"] = obs_reset_handles
                            self.buf_infos_dict[e]["reset_agent_mask"] = mask_reset_handles
                            self.buf_infos_dict[e]["reset_state"] = state_reset

                    self.waiting = False
                    return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), self.buf_trunctions.copy(), self.buf_infos_dict.copy()

                def render(self, mode=None):
                    return [env.render() for env in self.envs]

                def global_state(self):
                    return np.array([env.state() for env in self.envs])

                def global_state_one_env(self, e):
                    return np.array(self.envs[e].state())

                def agent_mask(self):
                    agent_mask = [np.ones([self.num_envs, n], dtype=np.bool) for n in self.n_agents]
                    for e, env in enumerate(self.envs):
                        mask = env.get_agent_mask()
                        for h, ids in enumerate(self.agent_ids):
                            agent_mask[h][e] = mask[ids]

                    return agent_mask



