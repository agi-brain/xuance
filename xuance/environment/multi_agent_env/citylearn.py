from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from xuance.environment.single_agent_env.citylearn import (
    _create_citylearn_env,
    _extract_district_kpis,
    _format_obs_vector,
    _read_current_net_load,
    _render_citylearn_dashboard,
    _rescale_from_unit_box,
    _resolve_max_episode_steps,
    _to_gymnasium_box,
    _unwrap_space_list,
)
from xuance.environment.utils import RawMultiAgentEnv


class CityLearn_MultiAgentEnv(RawMultiAgentEnv):
    """Multi-agent decentralized wrapper for CityLearn (``central_agent=False``)."""

    def __init__(self, config: Any):
        super().__init__()
        self.env, self.base_env, self.env_id = _create_citylearn_env(config, central_agent=False)
        self.render_mode = getattr(config, "render_mode", "rgb_array")
        self.evaluate_on_done = bool(getattr(config, "evaluate_on_done", False))
        self._initial_seed = getattr(config, "env_seed", None)

        obs_spaces, self._obs_is_list = _unwrap_space_list(self.env.observation_space)
        act_spaces, self._act_is_list = _unwrap_space_list(self.env.action_space)

        buildings = getattr(self.base_env, "buildings", None)
        if buildings and len(buildings) == len(obs_spaces):
            self.agents = [str(getattr(b, "name", f"building_{i}")) for i, b in enumerate(buildings)]
        else:
            self.agents = [f"building_{i}" for i in range(len(obs_spaces))]
        self.num_agents = len(self.agents)
        self.agent_groups = []

        self._raw_obs_dims: Dict[str, int] = {}
        self._raw_act_dims: Dict[str, int] = {}
        self._raw_act_lows: Dict[str, np.ndarray] = {}
        self._raw_act_highs: Dict[str, np.ndarray] = {}
        raw_obs_boxes: Dict[str, spaces.Box] = {}
        for i, k in enumerate(self.agents):
            obs_box = _to_gymnasium_box(obs_spaces[i])
            act_box = _to_gymnasium_box(act_spaces[i])
            raw_obs_boxes[k] = obs_box
            self._raw_obs_dims[k] = int(np.prod(obs_box.shape))
            self._raw_act_dims[k] = int(np.prod(act_box.shape))
            self._raw_act_lows[k] = act_box.low.reshape(-1).copy()
            self._raw_act_highs[k] = act_box.high.reshape(-1).copy()

        pad_cfg = getattr(config, "pad_spaces", None)
        self.pad_spaces = bool(getattr(config, "use_parameter_sharing", False)) if pad_cfg is None else bool(pad_cfg)

        if self.pad_spaces and self.agents:
            max_obs_dim = max(self._raw_obs_dims.values())
            max_act_dim = max(self._raw_act_dims.values())
            self.observation_space = {
                k: spaces.Box(
                    low=np.pad(raw_obs_boxes[k].low.reshape(-1), (0, max_obs_dim - self._raw_obs_dims[k])),
                    high=np.pad(raw_obs_boxes[k].high.reshape(-1), (0, max_obs_dim - self._raw_obs_dims[k])),
                    shape=(max_obs_dim,),
                    dtype=np.float32,
                )
                for k in self.agents
            }
            self.action_space = {
                k: spaces.Box(low=-1.0, high=1.0, shape=(max_act_dim,), dtype=np.float32) for k in self.agents
            }
        else:
            self.observation_space = {
                k: spaces.Box(
                    low=raw_obs_boxes[k].low.reshape(-1),
                    high=raw_obs_boxes[k].high.reshape(-1),
                    shape=(self._raw_obs_dims[k],),
                    dtype=np.float32,
                )
                for k in self.agents
            }
            self.action_space = {
                k: spaces.Box(low=-1.0, high=1.0, shape=(self._raw_act_dims[k],), dtype=np.float32) for k in self.agents
            }

        dim_state = sum(int(np.prod(self.observation_space[k].shape)) for k in self.agents)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_state,), dtype=np.float32)
        self.max_episode_steps = _resolve_max_episode_steps(self.base_env, config)

        self._episode_step = 0
        self._episode_reward = 0.0
        self._net_load_history: List[float] = []
        self._current_state = np.zeros(self.state_space.shape, dtype=np.float32)

    def _pack_obs_dict(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        obs_seq = raw_obs if self._obs_is_list else [raw_obs]
        return {
            k: _format_obs_vector(obs_seq[i], self.observation_space[k].shape[0])
            for i, k in enumerate(self.agents)
        }

    def evaluate_kpis(self) -> Dict[str, float]:
        """Compute district-level CityLearn KPIs for the current episode."""
        return _extract_district_kpis(self.base_env)

    def reset(self, **kwargs):
        """Reset the decentralized multi-agent CityLearn environment."""
        reset_kwargs = dict(kwargs)
        if "seed" not in reset_kwargs and self._initial_seed is not None:
            reset_kwargs["seed"] = int(self._initial_seed)
            self._initial_seed = None
        try:
            raw_obs, raw_info = self.env.reset(**reset_kwargs)
        except TypeError:
            raw_obs, raw_info = self.env.reset()

        obs_dict = self._pack_obs_dict(raw_obs)
        self._current_state = np.concatenate([obs_dict[k] for k in self.agents], axis=-1).astype(np.float32, copy=False)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._net_load_history = []
        init_load = _read_current_net_load(self.base_env)
        if init_load is not None:
            self._net_load_history.append(init_load)
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        return obs_dict, info

    def step(self, action_dict):
        """Execute per-building actions (rescaled from ``[-1, 1]``) and return MARL tuples."""
        env_actions: List[np.ndarray] = []
        for k in self.agents:
            act_vec = np.asarray(action_dict[k], dtype=np.float32).reshape(-1)[: self._raw_act_dims[k]]
            phys_act = _rescale_from_unit_box(act_vec, self._raw_act_lows[k], self._raw_act_highs[k])
            env_actions.append(phys_act)

        payload = env_actions if self._act_is_list else env_actions[0]
        raw_obs, raw_rews, raw_term, raw_trunc, raw_info = self.env.step(payload)

        obs_dict = self._pack_obs_dict(raw_obs)
        self._current_state = np.concatenate([obs_dict[k] for k in self.agents], axis=-1).astype(np.float32, copy=False)

        rew_seq = raw_rews if isinstance(raw_rews, (list, tuple, np.ndarray)) else [raw_rews] * self.num_agents
        reward_dict = {k: float(rew_seq[i]) for i, k in enumerate(self.agents)}
        self._episode_step += 1
        self._episode_reward += float(np.sum(list(reward_dict.values())))
        curr_load = _read_current_net_load(self.base_env)
        if curr_load is not None:
            self._net_load_history.append(curr_load)

        terminated = bool(raw_term)
        terminated_dict = {k: terminated for k in self.agents}
        truncated = bool(raw_trunc or (not terminated and self._episode_step >= self.max_episode_steps))
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        if (terminated or truncated) and self.evaluate_on_done:
            info["kpis"] = self.evaluate_kpis()

        return obs_dict, reward_dict, terminated_dict, truncated, info

    def state(self):
        """Return concatenated global observations across all buildings."""
        return self._current_state

    def agent_mask(self):
        """All building agents remain active throughout the episode."""
        return {k: True for k in self.agents}

    def avail_actions(self):
        """Continuous building control uses no discrete action mask."""
        return None

    def render(self, *args, **kwargs):
        """Render a headless RGB dashboard frame summarizing grid load and storage SoC."""
        return _render_citylearn_dashboard(
            self.base_env,
            episode_step=self._episode_step,
            max_episode_steps=self.max_episode_steps,
            episode_reward=self._episode_reward,
            net_load_history=self._net_load_history,
            title="XuanCe x CityLearn (MARL)",
        )

    def close(self):
        """Close the CityLearn environment."""
        self.env.close()
