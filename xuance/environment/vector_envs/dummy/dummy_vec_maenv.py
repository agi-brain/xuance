import numpy as np
from gym.spaces import Dict, Box
from xuance.common import space2shape, combined_shape
from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from operator import itemgetter


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

        self.teams = env.teams_info["names"]
        self.agents = env.agents
        self.n_agents_all = env.num_agents
        self.state_space = env.state_space  # Type: Box
        self.buf_state = np.zeros((self.num_envs,) + self.state_space.shape, dtype=self.state_space.dtype)
        self.buf_obs = [{} for _ in range(self.num_envs)]
        self.buf_avail_actions = [{} for _ in range(self.num_envs)]
        self.buf_info = [{} for _ in range(self.num_envs)]

        self.actions = None
        self.max_episode_length = env.max_episode_steps

    def reset(self):
        """Reset the vectorized environments."""
        for e in range(self.num_envs):
            self.buf_obs[e], self.buf_info[e] = self.envs[e].reset()
            self.buf_avail_actions[e] = self.buf_info[e]['avail_actions']
        return self.buf_obs.copy(), self.buf_info.copy()

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

        rew_dict = [{} for _ in self.envs]
        terminated_dict = [{} for _ in self.envs]
        truncated = [False for _ in self.envs]
        for e in range(self.num_envs):
            action_n = self.actions[e]
            self.buf_obs[e], rew_dict[e], terminated_dict[e], truncated[e], self.buf_info[e] = self.envs[e].step(action_n)
            self.buf_avail_actions[e] = self.buf_info[e]['avail_actions']
            if all(terminated_dict[e].values()) or truncated[e]:
                obs_reset_dict, info_reset = self.envs[e].reset()
                self.buf_info[e]["reset_obs"] = obs_reset_dict
                self.buf_info[e]["reset_avail_actions"] = info_reset['avail_actions']
        self.waiting = False
        return self.buf_obs.copy(), rew_dict, terminated_dict, truncated, self.buf_info.copy()

    def close_extras(self):
        """Closes the communication with subprocesses and joins the subprocesses."""
        self.closed = True
        for env in self.envs:
            try:
                env.close()
            except:
                pass

    def render(self, mode):
        return [env.render(mode) for env in self.envs]

