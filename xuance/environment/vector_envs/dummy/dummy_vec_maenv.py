import numpy as np
from gym.spaces import Dict, Box
from xuance.common import space2shape, combined_shape
from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError


class DummyVecMutliAgentEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    Parameters:
        env_fns – environment function.
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.teams = env.team_info["names"]
        self.agents = env.agents
        self.state_space = env.state_space  # Type: Box
        self.state_shape = self.state_space.shape  # Type: Tuple
        self.state_dtype = self.state_space.dtype  # Type: numpy.dtype
        obs_n_space = env.observation_space  # [Box(dim_o), Box(dim_o), ...] ----> dict

        self.keys, self.shapes, self.dtypes = obs_n_space_info(obs_n_space)  # self.keys: the keys for all agents.
        if isinstance(env.action_space[self.agents[0]], Box):
            self.act_dim = [env.action_space[keys].shape[0] for keys in self.agents]
        else:
            self.act_dim = [env.action_space[keys[0]].n for keys in self.agents]
        self.n_agent_all = len(self.keys)  # total number of agents
        self.num_agents = env.num_agents
        self.obs_shapes = [self.shapes[self.agents[0]] for h in
                           self.groups]  # suppose agents in one handle share a same observation space.
        self.obs_dtype = self.dtypes[self.keys[0]]

        # store data for current time step.
        # buffer of dict data
        self.buf_obs_dict = [{k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys} for _ in
                             range(self.num_envs)]
        self.buf_rews_dict = [{k: 0.0 for k in self.keys} for _ in range(self.num_envs)]
        self.buf_dones_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_trunctions_dict = [{k: False for k in self.keys} for _ in range(self.num_envs)]
        self.buf_infos_dict = [{} for _ in range(self.num_envs)]
        # buffer of numpy data
        self.buf_state = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
        self.buf_agent_mask = [np.ones([self.num_envs, n], dtype=np.bool_) for n in env.num_agents_group]
        self.buf_obs = [np.zeros((self.num_envs, n) + tuple(self.obs_shapes[h]), dtype=self.obs_dtype) for h, n in
                        enumerate(self.num_agents)]
        self.buf_rews = [np.zeros((self.num_envs, n, 1), dtype=np.float32) for n in env.num_agents_group]
        self.buf_dones = [np.ones((self.num_envs, n), dtype=np.bool_) for n in env.num_agents_group]
        self.buf_trunctions = [np.ones((self.num_envs, n), dtype=np.bool_) for n in env.num_agents_group]

        self.max_episode_length = env.max_cycles  # the max length of one episode.
        self.actions = None  # the actions to be executed.

    def empty_dict_buffers(self, i_env):
        """Reset the buffers for dictionary data."""
        self.buf_obs_dict[i_env] = {k: np.zeros(tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        self.buf_rews_dict[i_env] = {k: 0.0 for k in self.keys}
        self.buf_dones_dict[i_env] = {k: False for k in self.keys}
        self.buf_trunctions_dict[i_env] = {k: False for k in self.keys}
        self.buf_infos_dict[i_env] = {k: {} for k in self.keys}

    def reset(self):
        """Reset the vectorized environments."""
        for e in range(self.num_envs):
            obs, info = self.envs[e].reset()
            self.buf_obs_dict[e].update(obs)
            self.buf_infos_dict[e].update(info["infos"])
            for h, agent_keys_h in enumerate(self.agent_keys):
                self.buf_obs[h][e] = itemgetter(*agent_keys_h)(self.buf_obs_dict[e])
        return self.buf_obs.copy(), self.buf_infos_dict.copy()

    def step_async(self, actions):
        """Sends asynchronous step commands to each subprocess with the specified actions."""
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
        """
        Waits for the completion of asynchronous step operations and updates internal buffers with the received results.
        """
        if not self.waiting:
            raise NotSteppingError

        for e in range(self.num_envs):
            action_n = self.actions[e]
            o, r, d, t, info = self.envs[e].step(action_n)
            self.buf_state[e] = self.envs[e].state()
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
        """Sends a render command to each subprocess with the specified rendering mode."""
        return [env.render() for env in self.envs]

    def global_state(self):
        """Return the global state of the parallel environments."""
        return self.buf_state

    def agent_mask(self):
        """Return the agent mask."""
        return self.buf_agent_mask

    def available_actions(self):
        """Return an array representing available actions for each agent."""
        act_mask = [np.ones([self.num_envs, n, self.act_dim[h]], dtype=np.bool_) for h, n in enumerate(self.n_agents)]
        return np.array(act_mask)

    def close_extras(self):
        """Closes the communication with subprocesses and joins the subprocesses."""
        self.closed = True
        for env in self.envs:
            try: env.close()
            except: pass