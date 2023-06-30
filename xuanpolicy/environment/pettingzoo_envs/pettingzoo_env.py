from pettingzoo.utils.env import ParallelEnv
import numpy as np
import ctypes
import importlib
from xuanpolicy.environment.pettingzoo_envs import AGENT_NAME_DICT


class PettingZoo_Env(ParallelEnv):
    def __init__(self, env_name: str, env_id: str, seed: int, **kwargs):
        super(PettingZoo_Env, self).__init__()
        scenario = importlib.import_module('pettingzoo.' + env_name + '.' + env_id)
        self.env = scenario.parallel_env(continuous_actions=kwargs["continuous"],
                                         render_mode=kwargs["render_mode"])
        self.scenario_name = env_name + "." + env_id
        self.n_handles = len(AGENT_NAME_DICT[self.scenario_name])
        self.env.reset(seed)
        try:
            self.state_space = self.env.state_space
        except:
            self.state_space = None

        self.action_spaces = {k: self.env.action_space(k) for k in self.env.agents}
        self.observation_spaces = {k: self.env.observation_space(k) for k in self.env.agents}
        self.agents = self.env.agents
        self.n_agents_all = len(self.agents)

        self.handles = self.get_handles()

        self.agent_ids = [self.get_ids(h) for h in self.handles]
        self.n_agents = [self.get_num(h) for h in self.handles]

        # self.reward_range = env.reward_range
        self.metadata = self.env.metadata
        # self._warn_double_wrap()

        self._episode_step = 0
        self._episode_score = 0.0
        # assert self.spec.id in ENVIRONMENTS

        self.max_cycles = self.env.aec_env.env.env.max_cycles

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset()
        self._episode_step = 0
        self._episode_score = 0.0
        infos["episode_step"] = self._episode_step
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._episode_step += 1
        self._episode_score += rewards
        infos["episode_step"] = self._episode_step
        infos["episode_score"] = self._episode_score
        return observations, rewards, terminations, truncations, infos

    def state(self):
        try:
            return self.env.state()
        except:
            return None

    def get_num(self, handle):
        try:
            n = self.env.env.get_num(handle)
        except:
            n = len(self.get_ids(handle))
        return n

    def get_ids(self, handle):
        try:
            ids = self.env.env.get_agent_id(handle)
        except:
            agent_name = AGENT_NAME_DICT[self.scenario_name][handle.value]
            ids_handle = []
            for id, agent_key in enumerate(self.agents):
                if agent_name in agent_key:
                    ids_handle.append(id)
            ids = ids_handle
        return ids

    def get_agent_mask(self):
        if self.handles is None:
            return np.ones(self.n_agents_all, dtype=np.bool)  # all alive
        else:
            mask = np.zeros(self.n_agents_all, dtype=np.bool)  # all dead
            for handle in self.handles:
                try:
                    alive_ids = self.get_ids(handle)
                    mask[alive_ids] = True  # get alive agents
                except AttributeError("Cannot get the ids for alive agents!"):
                    return
        return mask

    def get_handles(self):
        if hasattr(self.env, 'handles'):
            return self.env.handles
        else:
            try:
                return self.env.env.get_handles()
            except:
                handles = [ctypes.c_int(h) for h in range(self.n_handles)]
                return handles