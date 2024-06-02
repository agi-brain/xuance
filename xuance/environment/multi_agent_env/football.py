"""
Football Benchmarks:
    - 11_vs_11_stochastic: A full 90 minutes football game (medium difficulty)
    - 11_vs_11_easy_stochastic: A full 90 minutes football game (easy difficulty)
    - 11_vs_11_hard_stochastic: A full 90 minutes football game (hard difficulty)

Football Academy - with a total of 11 scenarios
    - academy_empty_goal_close: Our player starts inside the box with the ball, and needs to score against an empty goal.
    - academy_empty_goal: Our player starts in the middle of the field with the ball, and needs to score against an empty goal.
    - academy_run_to_score: Our player starts in the middle of the field with the ball, and needs to score against an empty goal. Five opponent players chase ours from behind.
    - academy_run_to_score_with_keeper: Our player starts in the middle of the field with the ball, and needs to score against a keeper. Five opponent players chase ours from behind.
    - academy_pass_and_shoot_with_keeper: Two of our players try to score from the edge of the box, one is on the side with the ball, and next to a defender. The other is at the center, unmarked, and facing the opponent keeper.
    - academy_run_pass_and_shoot_with_keeper: Two of our players try to score from the edge of the box, one is on the side with the ball, and unmarked. The other is at the center, next to a defender, and facing the opponent keeper.
    - academy_3_vs_1_with_keeper: Three of our players try to score from the edge of the box, one on each side, and the other at the center. Initially, the player at the center has the ball and is facing the defender. There is an opponent keeper.
    - academy_corner: Standard corner-kick situation, except that the corner taker can run with the ball from the corner.
    - academy_counterattack_easy: 4 versus 1 counter-attack with keeper; all the remaining players of both teams run back towards the ball.
    - academy_counterattack_hard: 4 versus 2 counter-attack with keeper; all the remaining players of both teams run back towards the ball.
    - academy_single_goal_versus_lazy: Full 11 versus 11 games, where the opponents cannot move but they can only intercept the ball if it is close enough to them. Our center back defender has the ball at first.
"""
import numpy as np
import gfootball.env as football_env
from gym.spaces import Box
from gfootball.env import _apply_output_wrappers
from gym.spaces import MultiDiscrete, Discrete
from gfootball.env.football_env import FootballEnv
from gfootball.env import config as gf_config
from xuance.environment import RawMultiAgentEnv

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
    def __init__(self, config):
        write_goal_dumps = False
        dump_frequency = 1
        extra_players = None
        other_config_options = {}
        self.env_id = GFOOTBALL_ENV_ID[config.env_id]
        if config.test:
            write_full_episode_dumps = True
            self.render = True
            write_video = True
        else:
            write_full_episode_dumps = False
            self.render = False
            write_video = False
        self.n_agents = config.num_agent

        self.env = football_env.create_environment(
            env_name=self.env_id,
            stacked=config.use_stacked_frames,
            representation=config.obs_type,
            rewards=config.rewards_type,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=self.render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=config.videos_dir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=config.num_agent,
            number_of_right_players_agent_controls=config.num_adversary,
            channel_dimensions=(config.smm_width, config.smm_height),
            other_config_options=other_config_options
        ).unwrapped

        scenario_config = gf_config.Config({'level': self.env_id}).ScenarioConfig()
        players = [('agent:left_players=%d,right_players=%d' % (config.num_agent, config.num_adversary))]

        # Enable MultiAgentToSingleAgent wrapper?
        if scenario_config.control_all_players:
            if (config.num_agent in [0, 1]) and (config.num_adversary in [0, 1]):
                players = [('agent:left_players=%d,right_players=%d' %
                            (scenario_config.controllable_left_players if config.num_agent else 0,
                             scenario_config.controllable_right_players if config.num_adversary else 0))]

        if extra_players is not None:
            players.extend(extra_players)
        config_values = {
            'dump_full_episodes': write_full_episode_dumps,
            'dump_scores': write_goal_dumps,
            'players': players,
            'level': self.env_id,
            'tracesdir': config.videos_dir,
            'write_video': write_video,
        }
        config_values.update(other_config_options)
        c = gf_config.Config(config_values)
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


class GFootball_Env(RawMultiAgentEnv):
    """The wrapper of original football environment.

    Args:
        config: the SimpleNamespace variable that contains attributes to create an original env.
    """
    def __init__(self, config):
        super(GFootball_Env, self).__init__()
        env = football_raw_env(config)
        self.env = _apply_output_wrappers(env=env,
                                          rewards=config.rewards_type,
                                          representation=config.obs_type,
                                          channel_dimensions=(config.smm_width, config.smm_height),
                                          apply_single_agent_wrappers=(config.num_agent + config.num_adversary == 1),
                                          stacked=config.num_adversary)
        self.num_agents = config.num_agent
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.num_adversaries = config.num_adversary
        obs_shape_i = (self.env.observation_space.shape[-1], )
        self.observation_space = {k: Box(-np.inf, np.inf, obs_shape_i) for k in self.agents}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents)}
        self.max_episode_steps = self.env.unwrapped.observation()[0]['steps_left']
        self._episode_step = 0
        self.env.reset()
        state_shape = self.state().shape
        self.state_space = Box(-np.inf, np.inf, state_shape)

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *config, **kwconfig):
        """Get one-step frame."""
        return self.env.get_frame()

    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        return obs_dict, info

    def step(self, actions):
        """One-step transition of the environment.

        Args:
            actions: the actions for all agents.
        """
        actions_list = [int(actions[k]) for k in self.agents]
        obs, reward, terminated, truncated, info = self.env.step(actions_list)
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        reward_dict = {k: reward[i] for i, k in enumerate(self.agents)}
        terminated_dict = {k: terminated for k in self.agents}
        return obs_dict, reward_dict, terminated_dict, truncated, info

    def get_more_info(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["active"] = np.array([state[i]['active'] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info

    def state(self):
        """Get global state."""
        return np.array(self.env.env.state())

