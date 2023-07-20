import gfootball.env as football_env
from . import GFOOTBALL_ENV_ID
from gfootball.env.football_env import FootballEnv
from gfootball.env import config


class football_raw_env(FootballEnv):
    def __init__(self, args):
        write_goal_dumps = False
        write_full_episode_dumps = False
        extra_players = None
        write_video = False
        other_config_options = {}
        self.env_id = GFOOTBALL_ENV_ID[args.env_id]
        self.env = football_env.create_environment(
            env_name=self.env_id,
            stacked=args.use_stacked_frames,
            representation=args.obs_type,
            rewards=args.rewards_type,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=False,
            write_video=write_video,
            dump_frequency=1,
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
        return obs, reward, terminated, truncated, info
