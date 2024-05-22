from pettingzoo.utils.env import ParallelEnv
import numpy as np
import ctypes
import importlib

AGENT_NAME_DICT = {
    "mpe.simple_adversary_v3": ['adversary', 'agent'],
    "mpe.simple_crypto_v3": ['eve', 'alice', 'bob'],
    "mpe.simple_push_v3": ['adversary', 'agent'],
    "mpe.simple_reference_v3": ['agent'],
    "mpe.simple_speak_listener_v4": ['speaker', 'listener'],
    "mpe.simple_spread_v3": ['agent'],
    "mpe.simple_tag_v3": ['adversary', 'agent'],
    "mpe.simple_v3": ['agent'],
    "mpe.simple_world_comm_v3": ['adversary', 'agent'],
}

ATARI_ENVS_MARL = ['basketball_pong_v2', 'boxing_v1', 'combat_plan_v1', 'combat_tank_v1', 'double_dunk_v2',
                   'entombed_competitive_v2', 'entombed_cooperative_v2', 'flag_capture_v1', 'foozpong_v2',
                   'ice_hockey_v1', 'joust_v2', 'mario_bros_v2', 'maze_craze_v2', 'othello_v2', 'pong_v2',
                   'quadrapong_v3', 'space_invaders_v1', 'space_war_v1', 'surround_v1', 'tennis_v2',
                   'video_checkers_v3', 'volleyball_pong_v2', 'warlords_v2', 'wizard_of_wor_v2']
BUTTERFLY_ENVS_MARL = ['cooperative_pong_v3', 'knights_archers_zombies_v7', 'pistonball_v4', 'prison_v3',
                       'prospector_v4']
CLASSIC_ENVS_MARL = ['backgammon_v3', 'checkers_v3', 'chess_v4', 'connect_four_v3', 'dou_dizhu_v4', 'gin_rummy_v4',
                     'go_v5', 'hanabi_v4', 'leduc_holdem_v4', 'mahjong_v4', 'rps_v2', 'texas_holdem_no_limit_v5',
                     'texas_holdem_v4', 'tictactoe_v3', 'uno_v4']
SISL_ENVS_MARL = ['multiwalker_v7', 'pursuit_v3', 'waterworld_v3']


class PettingZoo_Env(ParallelEnv):
    """
    A wrapper for PettingZoo environments, provide a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning
    Parameters:
        env_name (str) – the name of the PettingZoo environment.
        env_id (str) – environment id.
        seed (int) – use to control randomness within the environment.
        kwargs (dict) – a variable-length keyword argument.
    """
    def __init__(self, env_name: str, env_id: str, seed: int, **kwargs):
        super(PettingZoo_Env, self).__init__()
        scenario = importlib.import_module('pettingzoo.' + env_name + '.' + env_id)
        self.continuous_actions = kwargs["continuous"]
        self.env = scenario.parallel_env(continuous_actions=self.continuous_actions,
                                         render_mode=kwargs["render_mode"])
        # self.env = scenario.env(continuous_actions=self.continuous_actions,
        #                                  render_mode=kwargs["render_mode"])
        self.scenario_name = env_name + "." + env_id
        self.n_handles = len(AGENT_NAME_DICT[self.scenario_name])
        self.side_names = AGENT_NAME_DICT[self.scenario_name]
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
        self.metadata = self.env.metadata
        try:
            self.max_cycles = self.env.unwrapped.max_cycles
        except:
            self.max_cycles = self.env.aec_env.env.env.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        observations, infos = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {"infos": infos,
                      "individual_episode_rewards": self.individual_episode_reward}
        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        if self.continuous_actions:
            for k, v in actions.items():
                actions[k] = np.clip(v, self.action_spaces[k].low, self.action_spaces[k].high)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"infos": infos,
                     "individual_episode_rewards": self.individual_episode_reward}
        return observations, rewards, terminations, truncations, step_info

    def state(self):
        """Returns the global state of the environment."""
        try:
            return np.array(self.env.state())
        except:
            return None

    def get_num(self, handle):
        """Returns the number of agents in a group."""
        try:
            n = self.env.env.get_num(handle)
        except:
            n = len(self.get_ids(handle))
        return n

    def get_ids(self, handle):
        """Returns the ids of all agents in the group."""
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
        """Create a boolean mask indicating which agents are currently alive."""
        if self.handles is None:
            return np.ones(self.n_agents_all, dtype=np.bool_)  # all alive
        else:
            mask = np.zeros(self.n_agents_all, dtype=np.bool_)  # all dead
            for handle in self.handles:
                try:
                    alive_ids = self.get_ids(handle)
                    mask[alive_ids] = True  # get alive agents
                except AttributeError("Cannot get the ids for alive agents!"):
                    return
        return mask

    def get_handles(self):
        """Returns all group handles in the environment."""
        if hasattr(self.env, 'handles'):
            return self.env.handles
        else:
            try:
                return self.env.env.get_handles()
            except:
                handles = [ctypes.c_int(h) for h in range(self.n_handles)]
                return handles
