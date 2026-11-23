import inspect
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces
from xuance.environment.utils import RawMultiAgentEnv
from xuance.environment.vector_envs.native import NativeVecMultiAgentEnv

try:
    import torch
except ImportError:
    torch = None

try:
    import vmas
except ImportError as err:
    vmas = None
    _VMAS_IMPORT_ERROR: Optional[BaseException] = err
else:
    _VMAS_IMPORT_ERROR = None


def _require_vmas():
    """Ensure the optional ``vmas`` package is available before creating envs."""
    if vmas is None:
        raise ImportError(
            "VMAS is not installed. Please install it via `pip install vmas` "
            "or `pip install xuance[vmas]`."
        ) from _VMAS_IMPORT_ERROR
    return vmas


def _to_gymnasium_space(space: Any, is_action: bool = False) -> spaces.Space:
    """Convert legacy ``gym.spaces`` or ``gymnasium.spaces`` objects to Gymnasium spaces."""
    cls_name = type(space).__name__
    if cls_name == "MultiDiscrete" or hasattr(space, "nvec"):
        if is_action:
            raise NotImplementedError(
                "VMAS MultiDiscrete action spaces are not supported by XuanCe MARL agents. "
                "Use `continuous_action=True` or a Discrete VMAS scenario."
            )
        return spaces.MultiDiscrete(np.asarray(space.nvec, dtype=np.int64))

    if isinstance(space, spaces.Box) or (hasattr(space, "low") and hasattr(space, "high")):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=tuple(space.shape), dtype=np.float32)

    if isinstance(space, spaces.Discrete) or hasattr(space, "n"):
        return spaces.Discrete(int(space.n))

    raise NotImplementedError(f"Unsupported VMAS space type: {type(space)!r}")


def _to_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert a PyTorch tensor or array-like value to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _extract_vmas_metadata(raw_env: Any) -> Tuple[List[str], Dict[str, spaces.Space], Dict[str, spaces.Space], spaces.Box]:
    """Extract ordered agent names, Gymnasium spaces, and concatenated global state space."""
    raw_obs_spaces = getattr(raw_env.observation_space, "spaces", raw_env.observation_space)
    raw_act_spaces = getattr(raw_env.action_space, "spaces", raw_env.action_space)

    raw_agents = getattr(raw_env, "agents", None)
    if raw_agents:
        agents = [a.name if hasattr(a, "name") else str(a) for a in raw_agents]
    else:
        agents = list(raw_obs_spaces.keys())

    observation_space = {k: _to_gymnasium_space(raw_obs_spaces[k], is_action=False) for k in agents}
    action_space = {k: _to_gymnasium_space(raw_act_spaces[k], is_action=True) for k in agents}

    dim_state = sum(int(np.prod(observation_space[k].shape)) for k in agents)
    state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_state,), dtype=np.float32)
    return agents, observation_space, action_space, state_space


def _build_groups_info(
    agent_groups: List[List[str]],
    observation_space: Dict[str, spaces.Space],
    action_space: Dict[str, spaces.Space],
) -> Dict[str, Any]:
    """Build the ``groups_info`` dictionary matching ``RawMultiAgentEnv.get_groups_info()``."""
    return {
        "num_groups": len(agent_groups),
        "agent_groups": agent_groups,
        "observation_space_groups": [{k: observation_space[k] for k in group} for group in agent_groups],
        "action_space_groups": [{k: action_space[k] for k in group} for group in agent_groups],
        "num_agents_groups": [len(group) for group in agent_groups],
    }


def _create_vmas_env(config: Any, num_envs: int):
    """Instantiate a raw VMAS batched environment from a XuanCe config object."""
    vmas_pkg = _require_vmas()
    scenario = getattr(config, "env_id", None) or "transport"
    vmas_device = getattr(config, "vmas_device", "cpu")
    continuous_actions = bool(getattr(config, "continuous_action", True))
    max_episode_steps = int(getattr(config, "max_episode_steps", 200))
    env_seed = getattr(config, "env_seed", None)
    scenario_kwargs = dict(getattr(config, "scenario_kwargs", None) or {})

    make_kwargs: Dict[str, Any] = {
        "scenario": scenario,
        "num_envs": num_envs,
        "device": vmas_device,
        "continuous_actions": continuous_actions,
        "max_steps": max_episode_steps,
        "seed": env_seed,
        "dict_spaces": True,
    }
    try:
        params = inspect.signature(vmas_pkg.make_env).parameters
        if "terminated_truncated" in params:
            make_kwargs["terminated_truncated"] = True
        if "clamp_actions" in params and continuous_actions:
            make_kwargs["clamp_actions"] = True
    except (TypeError, ValueError):
        pass

    make_kwargs.update(scenario_kwargs)
    return vmas_pkg.make_env(**make_kwargs), scenario, vmas_device, continuous_actions, max_episode_steps


def _pack_vmas_actions(
    actions_list: List[Dict[str, Any]],
    agents: List[str],
    action_space: Dict[str, spaces.Space],
    continuous_actions: bool,
    device: Any,
) -> Dict[str, Any]:
    """Pack per-environment action dicts into batched VMAS tensors or arrays."""
    num_envs = len(actions_list)
    packed: Dict[str, Any] = {}
    if continuous_actions:
        for k in agents:
            space = action_space[k]
            arr = np.asarray([actions_list[e][k] for e in range(num_envs)], dtype=np.float32)
            arr = arr.reshape((num_envs, *space.shape))
            arr = np.clip(arr, space.low, space.high)
            packed[k] = torch.as_tensor(arr, dtype=torch.float32, device=device) if torch is not None else arr
    else:
        for k in agents:
            arr = np.asarray(
                [int(np.asarray(actions_list[e][k]).item()) for e in range(num_envs)],
                dtype=np.int64,
            ).reshape(num_envs, 1)
            packed[k] = torch.as_tensor(arr, dtype=torch.long, device=device) if torch is not None else arr
    return packed


def _unpack_batch_obs_and_state(
    raw_obs: Dict[str, Any],
    agents: List[str],
    num_envs: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]:
    """Convert batched VMAS observation tensors to per-env dicts and concatenated global states."""
    obs_np = {k: _to_numpy(raw_obs[k], dtype=np.float32) for k in agents}
    obs_list = [{k: obs_np[k][e].copy() for k in agents} for e in range(num_envs)]
    state_batch = np.concatenate([obs_np[k].reshape(num_envs, -1) for k in agents], axis=-1).astype(
        np.float32, copy=False
    )
    state_list = [state_batch[e].copy() for e in range(num_envs)]
    return obs_list, state_list


def _unpack_single_reset_obs_and_state(
    reset_obs: Dict[str, Any],
    env_index: int,
    agents: List[str],
    observation_space: Dict[str, spaces.Space],
    num_envs: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract a single sub-environment observation dict and state from ``env.reset_at(e)``."""
    obs_e: Dict[str, np.ndarray] = {}
    for k in agents:
        arr = _to_numpy(reset_obs[k], dtype=np.float32)
        expected_shape = observation_space[k].shape
        if arr.shape == expected_shape:
            obs_e[k] = arr.copy()
        elif arr.shape == (1, *expected_shape):
            obs_e[k] = arr[0].copy()
        elif arr.shape == (num_envs, *expected_shape):
            obs_e[k] = arr[env_index].copy()
        else:
            obs_e[k] = arr[env_index].reshape(expected_shape).copy()
    state_e = np.concatenate([obs_e[k].reshape(-1) for k in agents], axis=-1).astype(np.float32, copy=False)
    return obs_e, state_e


def _build_avail_actions(
    agents: List[str],
    action_space: Dict[str, spaces.Space],
    continuous_actions: bool,
) -> Optional[Dict[str, np.ndarray]]:
    """Return all-True action masks for discrete scenarios, or ``None`` for continuous control."""
    if continuous_actions:
        return None
    return {k: np.ones(action_space[k].n, dtype=np.bool_) for k in agents}


def _parse_vmas_step_result(
    step_result: Tuple[Any, ...],
    episode_steps: np.ndarray,
    max_episode_steps: int,
    num_envs: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """Normalize 4-tuple or 5-tuple VMAS ``step()`` outputs into obs, rewards, term, and trunc arrays."""
    if len(step_result) == 5:
        raw_obs, raw_rews, raw_terminated, raw_truncated, _raw_info = step_result
        term_np = _to_numpy(raw_terminated, dtype=np.bool_).reshape(num_envs)
        trunc_np = _to_numpy(raw_truncated, dtype=np.bool_).reshape(num_envs)
    elif len(step_result) == 4:
        raw_obs, raw_rews, raw_dones, _raw_info = step_result
        dones_np = _to_numpy(raw_dones, dtype=np.bool_).reshape(num_envs)
        trunc_np = (episode_steps + 1) >= max_episode_steps
        term_np = np.logical_and(dones_np, np.logical_not(trunc_np))
    else:
        raise ValueError(f"Unexpected VMAS step() return tuple length: {len(step_result)}")
    return raw_obs, raw_rews, term_np, trunc_np


def _close_viewer(raw_env: Any) -> None:
    """Close any active pyglet viewer attached to a VMAS environment."""
    viewer = getattr(raw_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
        raw_env.viewer = None


class VMASVecMultiAgentEnv(NativeVecMultiAgentEnv):
    """Native batched XuanCe vectorized wrapper for VMAS multi-agent scenarios."""

    def __init__(self, config: Any):
        num_envs = max(1, int(getattr(config, "parallels", 1)))
        self.env, self.env_id, self.vmas_device, self.continuous_actions, self.max_episode_steps = _create_vmas_env(
            config, num_envs=num_envs
        )
        self.render_mode = getattr(config, "render_mode", "rgb_array")

        self.agents, self.observation_space, self.action_space, self.state_space = _extract_vmas_metadata(self.env)
        self.num_agents = len(self.agents)
        self.agent_groups: List[List[str]] = []
        self.env_info = {
            "state_space": self.state_space,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": self.agents,
            "num_agents": self.num_agents,
            "max_episode_steps": self.max_episode_steps,
        }
        self.groups_info = _build_groups_info(self.agent_groups, self.observation_space, self.action_space)

        super().__init__(num_envs, self.observation_space, self.action_space)
        self._init_buffers()
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_scores = [{k: 0.0 for k in self.agents} for _ in range(self.num_envs)]

    def _agent_mask_dict(self) -> Dict[str, bool]:
        return {k: True for k in self.agents}

    def _avail_actions_dict(self) -> Optional[Dict[str, np.ndarray]]:
        return _build_avail_actions(self.agents, self.action_space, self.continuous_actions)

    def reset(self):
        """Reset all parallel VMAS environments."""
        raw_obs = self.env.reset()
        obs_list, state_list = _unpack_batch_obs_and_state(raw_obs, self.agents, self.num_envs)
        self._episode_steps.fill(0)
        self._episode_scores = [{k: 0.0 for k in self.agents} for _ in range(self.num_envs)]

        for e in range(self.num_envs):
            avail_e = self._avail_actions_dict()
            self.buf_obs[e] = obs_list[e]
            self.buf_state[e] = state_list[e]
            self.buf_avail_actions[e] = avail_e
            self.buf_info[e] = {
                "episode_step": 0,
                "episode_score": self._episode_scores[e].copy(),
                "agent_mask": self._agent_mask_dict(),
                "avail_actions": avail_e,
                "state": self.buf_state[e],
            }

        return self.buf_obs.copy(), self.buf_info.copy()

    def step_wait(self):
        """Execute a batched VMAS step and handle partial sub-environment resets."""
        if not self.waiting:
            from xuance.environment.vector_envs.vector_env import NotSteppingError

            raise NotSteppingError

        batched_actions = _pack_vmas_actions(
            self.actions, self.agents, self.action_space, self.continuous_actions, self.vmas_device
        )
        step_result = self.env.step(batched_actions)
        raw_obs, raw_rews, term_np, trunc_np = _parse_vmas_step_result(
            step_result, self._episode_steps, self.max_episode_steps, self.num_envs
        )

        obs_list, state_list = _unpack_batch_obs_and_state(raw_obs, self.agents, self.num_envs)
        rews_np = {k: _to_numpy(raw_rews[k], dtype=np.float32).reshape(self.num_envs) for k in self.agents}
        self._episode_steps += 1

        rew_dicts: List[Dict[str, float]] = []
        terminated_dicts: List[Dict[str, bool]] = []
        truncated_list: List[bool] = []

        for e in range(self.num_envs):
            step_rew = {k: float(rews_np[k][e]) for k in self.agents}
            for k in self.agents:
                self._episode_scores[e][k] += step_rew[k]
            rew_dicts.append(step_rew)

            term_e = bool(term_np[e])
            trunc_e = bool(trunc_np[e] or (self._episode_steps[e] >= self.max_episode_steps))
            terminated_dicts.append({k: term_e for k in self.agents})
            truncated_list.append(trunc_e)

            avail_e = self._avail_actions_dict()
            self.buf_obs[e] = obs_list[e]
            self.buf_state[e] = state_list[e]
            self.buf_avail_actions[e] = avail_e

            info_e: Dict[str, Any] = {
                "episode_step": int(self._episode_steps[e]),
                "episode_score": self._episode_scores[e].copy(),
                "agent_mask": self._agent_mask_dict(),
                "avail_actions": avail_e,
                "state": self.buf_state[e],
            }

            if term_e or trunc_e:
                reset_raw_obs = self.env.reset_at(e)
                reset_obs_e, reset_state_e = _unpack_single_reset_obs_and_state(
                    reset_raw_obs, e, self.agents, self.observation_space, self.num_envs
                )
                self._episode_steps[e] = 0
                self._episode_scores[e] = {k: 0.0 for k in self.agents}
                info_e["reset_obs"] = reset_obs_e
                info_e["reset_state"] = reset_state_e
                info_e["reset_avail_actions"] = self._avail_actions_dict()

            self.buf_info[e] = info_e

        self.waiting = False
        return self.buf_obs.copy(), rew_dicts, terminated_dicts, truncated_list, self.buf_info.copy()

    def render(self, mode: Optional[str] = None):
        """Render each sub-environment using VMAS's viewer."""
        render_mode = mode or self.render_mode or "rgb_array"
        return [self.env.render(mode=render_mode, env_index=e) for e in range(self.num_envs)]

    def close_extras(self):
        """Release any VMAS pyglet viewer resources."""
        _close_viewer(self.env)


class VMAS_Env(RawMultiAgentEnv):
    """Single-environment VMAS wrapper for fallback with DummyVec/SubprocVecMultiAgentEnv."""

    @classmethod
    def make_vec_env(cls, config: Any) -> VMASVecMultiAgentEnv:
        """Factory used by ``make_envs`` when ``vectorize='NativeVecMultiAgentEnv'``."""
        return VMASVecMultiAgentEnv(config)

    def __init__(self, config: Any):
        super().__init__()
        self.env, self.env_id, self.vmas_device, self.continuous_actions, self.max_episode_steps = _create_vmas_env(
            config, num_envs=1
        )
        self.render_mode = getattr(config, "render_mode", "rgb_array")
        self.agents, self.observation_space, self.action_space, self.state_space = _extract_vmas_metadata(self.env)
        self.num_agents = len(self.agents)
        self.agent_groups = []
        self._episode_step = 0
        self._current_state = np.zeros(self.state_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs = self.env.reset()
        obs_list, state_list = _unpack_batch_obs_and_state(raw_obs, self.agents, num_envs=1)
        self._episode_step = 0
        self._current_state = state_list[0]
        return obs_list[0], {}

    def step(self, action_dict):
        batched_actions = _pack_vmas_actions(
            [action_dict], self.agents, self.action_space, self.continuous_actions, self.vmas_device
        )
        step_result = self.env.step(batched_actions)
        raw_obs, raw_rews, term_np, trunc_np = _parse_vmas_step_result(
            step_result, np.array([self._episode_step], dtype=np.int32), self.max_episode_steps, num_envs=1
        )
        term = bool(term_np[0])
        trunc = bool(trunc_np[0])

        obs_list, state_list = _unpack_batch_obs_and_state(raw_obs, self.agents, num_envs=1)
        self._current_state = state_list[0]
        self._episode_step += 1
        trunc = bool(trunc or (self._episode_step >= self.max_episode_steps))

        rewards = {k: float(_to_numpy(raw_rews[k], dtype=np.float32).reshape(1)[0]) for k in self.agents}
        terminated = {k: term for k in self.agents}
        return obs_list[0], rewards, terminated, trunc, {}

    def state(self):
        return self._current_state

    def agent_mask(self):
        return {k: True for k in self.agents}

    def avail_actions(self):
        return _build_avail_actions(self.agents, self.action_space, self.continuous_actions)

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else self.render_mode) or "rgb_array"
        return self.env.render(mode=mode, env_index=0)

    def close(self):
        _close_viewer(self.env)
