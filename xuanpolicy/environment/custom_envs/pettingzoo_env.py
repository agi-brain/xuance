import gym
import numpy as np
import ctypes

MPE_N_HANDLE_DICT = {
    "mpe.simple_adversary_v3": 2,
    "mpe.simple_crypto_v3": 3,
    "mpe.simple_push_v3": 2,
    "mpe.simple_reference_v3": 1,
    "mpe.simple_speak_listener_v4": 2,
    "mpe.simple_spread_v3": 1,
    "mpe.simple_tag_v3": 2,
    "mpe.simple_v3": 1,
    "mpe.simple_world_comm_v3": 1,
}

MPE_AGENT_NAME_DICT = {
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
MAGENT_ENVS_LARGE_SCALE_MARL = ['adversarial_pursuit_v3', 'battlefield_v3', 'battle_v3', 'combined_arms_v5',
                                'gather_v3', 'magent_env', 'tiger_deer_v3']
MPE_ENVS_MARL = ['simple_adversary_v2', 'simple_crypto_v2', 'simple_push_v2', 'simple_reference_v2',
                 'simple_speaker_listener_v3', 'simple_spread_v2', 'simple_tag_v2', 'simple_v2', 'simple_world_comm_v2']
SISL_ENVS_MARL = ['multiwalker_v7', 'pursuit_v3', 'waterworld_v3']
PETTINGZOO_ENVS = ['atari', 'butterfly', 'classic', 'magent', 'mpe', 'sisl']


class PettingZooWrapper(gym.Wrapper):
    def __init__(self, env, scenario_name):
        # super(PettingZooWrapper, self).__init__(env)
        self.env = env
        self.scenario_name = scenario_name
        self.env.reset()

        try:
            self.state_space = self.env.state_space
        except:
            self.state_space = None
        self.action_spaces = self.env.action_spaces
        self.observation_spaces = self.env.observation_spaces
        self.agents = self.env.action_spaces.keys()
        self.n_agents_all = len(self.agents)

        self.handles = self.get_handles()

        self.agent_ids = [self.get_ids(h) for h in self.handles]
        self.n_agents = [self.get_num(h) for h in self.handles]

        # self.reward_range = env.reward_range
        self.metadata = env.metadata
        # self._warn_double_wrap()

        self.episode_length = 0
        # assert self.spec.id in ENVIRONMENTS

        self.max_cycles = self.env.aec_env.env.env.max_cycles

    def step(self, action):
        self.episode_length += 1
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        return observations, rewards, terminations, truncations, infos

    def reset(self):
        self.episode_length = 0
        return self.env.reset()

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
            agent_name = MPE_AGENT_NAME_DICT[self.scenario_name][handle.value]
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
        try: return self.env.handles
        except:
            try: return self.env.env.get_handles()
            except:
                handles = [ctypes.c_int(h) for h in range(MPE_N_HANDLE_DICT[self.scenario_name])]
                # print("env.handles is None, now is set as: ", handles)
                return handles