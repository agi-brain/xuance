import numpy as np
from xuance.environment import RawMultiAgentEnv
from gymnasium.spaces import Box, Discrete

try:
    import sys
    for module_name in list(sys.modules.keys()):
        if 'smac.' in module_name:
            del sys.modules[module_name]
    from pysc2.maps import lib
    
    original_get_maps = lib.get_maps
    def patched_get_maps():
        maps = {}
        for mp in lib.Map.all_subclasses():
            if mp.filename or mp.battle_net:
                map_name = mp.__name__
                maps[map_name] = mp
        return maps
    lib.get_maps = patched_get_maps
except:
    pass

try:
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
except ImportError:
    pass


class SMACv2_Env(RawMultiAgentEnv):
    """
    The implementation of SMACv2 environments, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
    """

    def __init__(self, config):
        super(SMACv2_Env, self).__init__()
        
        self.capability_config = self._build_capability_config(config)
        
        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=self.capability_config,
            map_name=config.env_id,
            debug=getattr(config, 'debug', False),
            conic_fov=getattr(config, 'conic_fov', False),
            use_unit_ranges=getattr(config, 'use_unit_ranges', True),
            min_attack_range=getattr(config, 'min_attack_range', 2),
            obs_own_pos=getattr(config, 'obs_own_pos', True),
            fully_observable=getattr(config, 'fully_observable', False),
            step_mul=getattr(config, 'step_mul', 8),
            move_amount=getattr(config, 'move_amount', 2),
            difficulty=getattr(config, 'difficulty', "7"),
            reward_sparse=getattr(config, 'reward_sparse', False),
            reward_only_positive=getattr(config, 'reward_only_positive', True),
            reward_death_value=getattr(config, 'reward_death_value', 10),
            reward_win=getattr(config, 'reward_win', 200),
            reward_defeat=getattr(config, 'reward_defeat', 0),
            reward_negative_scale=getattr(config, 'reward_negative_scale', 0.5),
            reward_scale=getattr(config, 'reward_scale', True),
            reward_scale_rate=getattr(config, 'reward_scale_rate', 20),
            obs_all_health=getattr(config, 'obs_all_health', True),
            obs_own_health=getattr(config, 'obs_own_health', True),
            obs_last_action=getattr(config, 'obs_last_action', False),
            obs_pathing_grid=getattr(config, 'obs_pathing_grid', False),
            obs_terrain_height=getattr(config, 'obs_terrain_height', False),
            obs_instead_of_state=getattr(config, 'obs_instead_of_state', False),
            obs_timestep_number=getattr(config, 'obs_timestep_number', False),
            state_last_action=getattr(config, 'state_last_action', True),
            state_timestep_number=getattr(config, 'state_timestep_number', False),
        )
        self.env_info = self.env.get_env_info()

        self.num_agents = self.env_info['n_agents']
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_groups = {"agent": self.agents}
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(self.env_info['state_shape'],))
        self.observation_space = {k: Box(low=-np.inf, high=np.inf, shape=(self.env_info['obs_shape'],))
                                  for k in self.agents}
        self.action_space = {k: Discrete(n=self.env_info['n_actions']) for k in self.agents}
        self.env.reset()
        self.max_episode_steps = self.env_info['episode_limit']
        self._episode_step = 0

    def _build_capability_config(self, config):
        capability_config = {
            "n_units": getattr(config, 'n_units', 10),
            "n_enemies": getattr(config, 'n_enemies', 11),
        }
        
        if hasattr(config, 'team_gen'):
            capability_config["team_gen"] = config.team_gen
        else:
            capability_config["team_gen"] = {
                "dist_type": "weighted_teams",
                "unit_types": getattr(config, 'unit_types', ["marine", "marauder", "medivac"]),
                "weights": getattr(config, 'unit_weights', [0.45, 0.45, 0.1]),
                "observe": True,
                "exception_unit_types": getattr(config, 'exception_unit_types', ["medivac"]),
            }
        
        if hasattr(config, 'start_positions'):
            capability_config["start_positions"] = config.start_positions
        else:
            capability_config["start_positions"] = {
                "dist_type": getattr(config, 'start_dist_type', "surrounded_and_reflect"),
                "p": getattr(config, 'start_p', 0.5),
                "map_x": getattr(config, 'map_x', 32),
                "map_y": getattr(config, 'map_y', 32),
            }
        
        return capability_config

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.env_info["n_agents"],
                'max_episode_steps': self.max_episode_steps,
                'num_enemies': self.env.n_enemies}

    def get_groups_info(self):
        return self.agent_groups

    def reset(self):
        obs, _ = self.env.reset()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}
        self._episode_step = 0
        info = {}
        return obs_dict, info

    def step(self, actions):
        actions_list = [actions[key] for key in self.agents]
        reward, terminated, info = self.env.step(actions_list)
        if info == {}:
            info = {'battle_won': 0,
                    'dead_allies': 0,
                    'dead_enemies': 0}
        reward_dict = {k: reward for k in self.agents}
        terminated_dict = {k: terminated for k in self.agents}
        obs = self.env.get_obs()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}

        step_info = info
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False
        return obs_dict, reward_dict, terminated_dict, truncated, step_info

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def state(self):
        return self.env.get_state()

    def agent_mask(self):
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        actions_mask_list = self.env.get_avail_actions()
        return {key: actions_mask_list[index] for index, key in enumerate(self.agents)}
