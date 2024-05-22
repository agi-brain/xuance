import gfootball.env as football_env
from gfootball.env.football_env import FootballEnv
from gfootball.env import config
import numpy as np

GFOOTBALL_ENV_ID = {
    "1v1": "1_vs_1_easy",
    "5v5": "5_vs_5",
    "11v11_competition": "11_vs_11_competition",
    "11v11_kaggle": "11_vs_11_kaggle",
    "11v11": "11_vs_11_stochastic",
    "11v11_easy": "11_vs_11_easy_stochastic",
    "11v11_hard": "11_vs_11_hard_stochastic",
    "3v1": "academy_3_vs_1_with_keeper",
    "corner": "academy_corner",
    "ca_easy": "academy_counterattack_easy",
    "ca_hard": "academy_counterattack_hard",
    "eg": "academy_empty_goal",
    "eg_close": "academy_empty_goal_close",
    "psk": "academy_pass_and_shoot_with_keeper",
    "rpsk": "academy_run_pass_and_shoot_with_keeper",
    "rs": "academy_run_to_score",
    "rsk": "academy_run_to_score_with_keeper",
    "single_gvl": "academy_single_goal_versus_lazy",
}


class football_raw_env(FootballEnv):
    def __init__(self, args):
        write_goal_dumps = False
        dump_frequency = 1
        extra_players = None
        other_config_options = {}
        self.env_id = GFOOTBALL_ENV_ID[args.env_id]
        if args.test:
            write_full_episode_dumps = True
            self.render = True
            write_video = True
        else:
            write_full_episode_dumps = False
            self.render = False
            write_video = False
        self.n_agents = args.num_agent

        self.env = football_env.create_environment(
            env_name=self.env_id,
            stacked=args.use_stacked_frames,
            representation=args.obs_type,
            rewards=args.rewards_type,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=self.render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=args.videos_dir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=args.num_agent,
            number_of_right_players_agent_controls=args.num_adversary,
            channel_dimensions=(args.smm_width, args.smm_height),
            other_config_options=other_config_options
        ).unwrapped

        scenario_config = config.Config({'level': self.env_id}).ScenarioConfig()
        players = [('agent:left_players=%d,right_players=%d' % (args.num_agent, args.num_adversary))]

        # Enable MultiAgentToSingleAgent wrapper?
        if scenario_config.control_all_players:
            if (args.num_agent in [0, 1]) and (args.num_adversary in [0, 1]):
                players = [('agent:left_players=%d,right_players=%d' %
                            (scenario_config.controllable_left_players if args.num_agent else 0,
                             scenario_config.controllable_right_players if args.num_adversary else 0))]

        if extra_players is not None:
            players.extend(extra_players)
        config_values = {
            'dump_full_episodes': write_full_episode_dumps,
            'dump_scores': write_goal_dumps,
            'players': players,
            'level': self.env_id,
            'tracesdir': args.videos_dir,
            'write_video': write_video,
        }
        config_values.update(other_config_options)
        c = config.Config(config_values)
        super(football_raw_env, self).__init__(c)

    def reset(self):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        truncated = False
        global_reward = np.sum(reward)
        reward_n = np.array([global_reward] * self.n_agents)
        return obs, reward_n, terminated, truncated, info

    def get_frame(self):
        original_obs = self.env._env._observation
        frame = original_obs["frame"] if self.render else []
        return frame

    def state(self):
        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            elif type(obj) == int:
                return np.array([obj])
            else:
                return obj.flatten()

        original_obs = self.env._env._observation
        state = []
        for k, v in original_obs.items():
            if k == "ball_owned_team":
                if v == -1:
                    state.extend([1, 0, 0])
                elif v == 0:
                    state.extend([0, 1, 0])
                else:
                    state.extend([0, 0, 1])
            elif k == "game_mode":
                game_mode = [0] * 7
                game_mode[v] = 1
                state.extend(game_mode)
            elif k == "frame":
                pass
            else:
                state.extend(do_flatten(v))
        return state
