import numpy as np
from pettingzoo.utils.env import ParallelEnv
from xuance.environment.pettingzoo.pettingzoo_env import PettingZoo_Env
from xuance.environment.magent2 import AGENT_NAME_DICT
import importlib
from gymnasium.spaces.box import Box


class MAgent_Env(PettingZoo_Env, ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, env_id: str, seed: int, **kwargs):
        scenario = importlib.import_module('xuance.environment.magent2.environments.' + env_id)

        if env_id in ["adversarial_pursuit_v4"]:
            kwargs['minimap_mode'] = False
            kwargs['tag_penalty'] = -0.2
        if env_id in ["battle_v4", "battlefield_v4", "combined_arms_v6"]:
            kwargs['step_reward'] = -0.005
            kwargs['dead_penalty'] = -0.1
            kwargs['attack_penalty'] = -0.1
            kwargs['attack_opponent_reward'] = 0.2
        if env_id in ["gather_v4"]:
            kwargs['step_reward'] = -0.01
            kwargs['dead_penalty'] = -1
            kwargs['attack_peanlty'] = -0.1
            kwargs['attack_food_reward'] = 0.5
        if env_id in ["tiger_deer_v3"]:
            kwargs['tiger_step_recover'] = -0.1
            kwargs['deer_attacked'] = -0.1

        self.env = scenario.env(**kwargs).unwrapped
        self.scenario_name = 'magent2.' + env_id
        self.n_handles = len(self.env.handles)
        self.side_names = AGENT_NAME_DICT[env_id]
        self.env.reset(seed)

        self.state_space = self.env.state_space
        self.observation_spaces = {}
        for k in self.env.agents:
            obs_space = self.env.observation_spaces[k]
            self.observation_spaces.update({
                k: Box(np.min(obs_space.low), np.max(obs_space.high), [np.prod(obs_space.shape)], obs_space.dtype, seed)
            })
        self.action_spaces = {k: self.env.action_spaces[k] for k in self.env.agents}
        self.agents = self.env.agents
        self.n_agents_all = len(self.agents)

        self.handles = self.env.handles

        self.agent_ids = [self.env.env.get_agent_id(h) for h in self.handles]
        self.n_agents = [self.env.env.get_num(h) for h in self.handles]

        self.metadata = self.env.metadata
        self.max_cycles = self.env.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

    def reset(self, seed=None, option=None):
        observations = self.env.reset(seed, option)
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
            observations[agent_key] = observations[agent_key].reshape([-1])
        reset_info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward
        }
        return observations, reset_info

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
            observations[k] = observations[k].reshape([-1])
        step_info = {"infos": infos,
                     "individual_episode_rewards": self.individual_episode_reward}
        return observations, rewards, terminations, truncations, step_info
