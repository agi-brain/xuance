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
PETTINGZOO_ENVIRONMENTS = ['atari', 'butterfly', 'classic', 'mpe', 'sisl']
