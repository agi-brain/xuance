import inspect
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from gymnasium import spaces

from xuance.environment.utils import RawEnvironment

try:
    import citylearn
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import NormalizedObservationWrapper
except ImportError as err:
    citylearn = None
    CityLearnEnv = None
    NormalizedObservationWrapper = None
    _CITYLEARN_IMPORT_ERROR: Optional[BaseException] = err
else:
    _CITYLEARN_IMPORT_ERROR = None


DEFAULT_CITYLEARN_DATASET = "citylearn_challenge_2022_phase_1"


def _require_citylearn():
    """Ensure the optional ``citylearn`` package is available before creating envs."""
    if citylearn is None or CityLearnEnv is None:
        raise ImportError(
            "CityLearn is not installed. Please install it via `pip install citylearn` "
            "or `pip install \"xuance[citylearn]\"`."
        ) from _CITYLEARN_IMPORT_ERROR
    return CityLearnEnv


def _to_gymnasium_box(space: Any, override_low: Optional[float] = None, override_high: Optional[float] = None) -> spaces.Box:
    """Convert a Box-like space to a float32 ``gymnasium.spaces.Box``."""
    shape = tuple(int(dim) for dim in space.shape)
    if override_low is not None:
        low = np.full(shape, override_low, dtype=np.float32)
    else:
        low = np.asarray(space.low, dtype=np.float32).reshape(shape)
    if override_high is not None:
        high = np.full(shape, override_high, dtype=np.float32)
    else:
        high = np.asarray(space.high, dtype=np.float32).reshape(shape)
    return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)


def _unwrap_space_list(raw_space: Any) -> Tuple[List[Any], bool]:
    """Normalize CityLearn's list-of-Box spaces (or single Box) to a list and is_list flag."""
    if isinstance(raw_space, (list, tuple)):
        return list(raw_space), True
    if hasattr(raw_space, "spaces") and isinstance(getattr(raw_space, "spaces"), (list, tuple)):
        return list(raw_space.spaces), True
    return [raw_space], False


def _rescale_from_unit_box(action: Any, raw_low: np.ndarray, raw_high: np.ndarray) -> np.ndarray:
    """Map an action in ``[-1, 1]`` linearly onto ``[raw_low, raw_high]``."""
    arr = np.asarray(action, dtype=np.float32).reshape(raw_low.shape)
    clipped = np.clip(arr, -1.0, 1.0)
    scaled = raw_low + 0.5 * (clipped + 1.0) * (raw_high - raw_low)
    return np.clip(scaled, raw_low, raw_high).astype(np.float32, copy=False)


def _format_obs_vector(raw_obs: Any, target_dim: int) -> np.ndarray:
    """Convert a CityLearn observation sequence into a contiguous float32 1-D array."""
    arr = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
    if arr.shape[0] == target_dim:
        return arr.copy()
    if arr.shape[0] < target_dim:
        padded = np.zeros((target_dim,), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:target_dim].copy()


def _create_citylearn_env(config: Any, central_agent: bool) -> Tuple[Any, Any, Any]:
    """Instantiate a raw (and optionally observation-normalized) CityLearn environment."""
    env_cls = _require_citylearn()
    schema = getattr(config, "schema", None) or getattr(config, "env_id", None) or DEFAULT_CITYLEARN_DATASET
    episode_time_steps = getattr(config, "episode_time_steps", None)
    env_seed = getattr(config, "env_seed", None)
    citylearn_render_mode = getattr(config, "citylearn_render_mode", "none")
    extra_kwargs = dict(getattr(config, "citylearn_kwargs", None) or getattr(config, "scenario_kwargs", None) or {})

    make_kwargs: Dict[str, Any] = dict(extra_kwargs)
    make_kwargs["central_agent"] = central_agent
    if episode_time_steps is not None:
        make_kwargs["episode_time_steps"] = int(episode_time_steps)
    if env_seed is not None and "random_seed" not in make_kwargs:
        make_kwargs["random_seed"] = int(env_seed)

    try:
        params = inspect.signature(env_cls.__init__).parameters
        if "render_mode" in params:
            make_kwargs.setdefault("render_mode", citylearn_render_mode)
    except (TypeError, ValueError):
        make_kwargs.setdefault("render_mode", citylearn_render_mode)

    raw_env = env_cls(schema, **make_kwargs)
    normalize_obs = bool(getattr(config, "normalize_obs", True))
    if normalize_obs and NormalizedObservationWrapper is not None and not getattr(raw_env, "_is_stub", False):
        wrapped_env = NormalizedObservationWrapper(raw_env)
    else:
        wrapped_env = raw_env
    base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
    return wrapped_env, base_env, schema


def _resolve_max_episode_steps(base_env: Any, config: Any) -> int:
    """Compute finite transition horizon matching CityLearn's ``time_steps - 1`` stepping count."""
    time_steps = getattr(base_env, "time_steps", None)
    if time_steps is not None and int(time_steps) > 1:
        raw_horizon = int(time_steps) - 1
    else:
        raw_horizon = int(getattr(config, "max_episode_steps", 24) or 24)
    cfg_max_steps = getattr(config, "max_episode_steps", None)
    if cfg_max_steps is not None:
        return max(1, min(raw_horizon, int(cfg_max_steps)))
    return max(1, raw_horizon)


def _extract_district_kpis(base_env: Any) -> Dict[str, float]:
    """Extract district-level KPI scalars from ``base_env.evaluate()``."""
    evaluate_fn = getattr(base_env, "evaluate", None)
    if not callable(evaluate_fn):
        return {}
    try:
        kpis_df = evaluate_fn()
    except Exception:
        return {}
    if kpis_df is None or not hasattr(kpis_df, "iterrows"):
        return {}

    kpis: Dict[str, float] = {}
    has_level = hasattr(kpis_df, "columns") and "level" in kpis_df.columns
    for _, row in kpis_df.iterrows():
        if has_level and str(row["level"]) != "district":
            continue
        name = str(row["cost_function"])
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            kpis[name] = value
    return kpis


def _read_current_net_load(base_env: Any) -> Optional[float]:
    """Safely read the latest district net electricity consumption from ``base_env``."""
    series = getattr(base_env, "net_electricity_consumption", None)
    if series is None or len(series) == 0:
        return None
    time_step = int(getattr(base_env, "time_step", len(series) - 1))
    idx = min(max(0, time_step - 1 if len(series) == time_step else time_step), len(series) - 1)
    try:
        val = float(series[idx])
    except (TypeError, ValueError, IndexError):
        return None
    return val if np.isfinite(val) else None


def _read_building_socs(base_env: Any) -> List[Tuple[str, float]]:
    """Safely extract per-building storage State-of-Charge values in ``[0, 1]``."""
    buildings = getattr(base_env, "buildings", None) or []
    time_step = int(getattr(base_env, "time_step", 0))
    socs: List[Tuple[str, float]] = []
    for idx_b, b in enumerate(buildings):
        name = str(getattr(b, "name", f"B{idx_b + 1}"))
        soc_val = 0.0
        for storage_attr in ("electrical_storage", "dhw_storage", "cooling_storage", "heating_storage"):
            storage = getattr(b, storage_attr, None)
            soc_arr = getattr(storage, "soc", None) if storage is not None else None
            if soc_arr is not None and len(soc_arr) > 0:
                t_idx = min(max(0, time_step - 1 if time_step > 0 else 0), len(soc_arr) - 1)
                try:
                    candidate = float(soc_arr[t_idx])
                    if np.isfinite(candidate):
                        soc_val = float(np.clip(candidate, 0.0, 1.0))
                        break
                except (TypeError, ValueError, IndexError):
                    pass
        socs.append((name, soc_val))
    return socs


def _render_citylearn_dashboard(
    base_env: Any,
    episode_step: int,
    max_episode_steps: int,
    episode_reward: float,
    net_load_history: List[float],
    title: str = "XuanCe x CityLearn",
) -> np.ndarray:
    """Render a lightweight headless RGB dashboard summarizing grid load and storage SoC."""
    height, width = 360, 640
    frame = np.full((height, width, 3), fill_value=(18, 24, 38), dtype=np.uint8)

    # Header bar
    cv2.rectangle(frame, (0, 0), (width, 46), (28, 37, 56), -1)
    cv2.putText(frame, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 242, 255), 2, cv2.LINE_AA)
    stats_txt = f"Step {episode_step}/{max_episode_steps}  |  Return {episode_reward:.2f}"
    cv2.putText(frame, stats_txt, (320, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 210, 255), 1, cv2.LINE_AA)

    # Left panel: Net grid load trace
    px0, py0, pw, ph = 20, 66, 360, 264
    cv2.rectangle(frame, (px0, py0), (px0 + pw, py0 + ph), (26, 33, 50), -1)
    cv2.rectangle(frame, (px0, py0), (px0 + pw, py0 + ph), (55, 70, 98), 1)
    cv2.putText(frame, "District Net Electricity Load (kWh)", (px0 + 12, py0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 245), 1, cv2.LINE_AA)

    plot_x0, plot_y0 = px0 + 16, py0 + 42
    plot_w, plot_h = pw - 32, ph - 64

    baseline_series = getattr(base_env, "net_electricity_consumption_without_storage", None)
    ctrl_vals = list(net_load_history) if net_load_history else []
    base_vals: List[float] = []
    if baseline_series is not None and len(baseline_series) > 0 and ctrl_vals:
        n_pts = min(len(ctrl_vals), len(baseline_series))
        base_vals = [float(baseline_series[i]) for i in range(n_pts) if np.isfinite(baseline_series[i])]

    all_vals = ctrl_vals + base_vals
    if all_vals:
        vmin = min(all_vals)
        vmax = max(all_vals)
        if vmax - vmin < 1e-3:
            vmin -= 1.0
            vmax += 1.0
        pad = 0.1 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        def _to_pt(idx: int, total: int, val: float) -> Tuple[int, int]:
            denom_x = max(1, total - 1)
            x = plot_x0 + int(round((idx / denom_x) * plot_w))
            y_norm = (val - vmin) / max(1e-6, vmax - vmin)
            y = plot_y0 + plot_h - int(round(y_norm * plot_h))
            return x, y

        if vmin <= 0.0 <= vmax:
            _, zero_y = _to_pt(0, 2, 0.0)
            cv2.line(frame, (plot_x0, zero_y), (plot_x0 + plot_w, zero_y), (70, 82, 105), 1, cv2.LINE_AA)

        if len(base_vals) >= 2:
            pts_b = [_to_pt(i, max(len(ctrl_vals), 2), v) for i, v in enumerate(base_vals)]
            for i in range(len(pts_b) - 1):
                cv2.line(frame, pts_b[i], pts_b[i + 1], (120, 130, 150), 1, cv2.LINE_AA)

        if len(ctrl_vals) >= 2:
            pts_c = [_to_pt(i, len(ctrl_vals), v) for i, v in enumerate(ctrl_vals)]
            for i in range(len(pts_c) - 1):
                cv2.line(frame, pts_c[i], pts_c[i + 1], (56, 189, 248), 2, cv2.LINE_AA)
        elif len(ctrl_vals) == 1:
            pt = _to_pt(0, 2, ctrl_vals[0])
            cv2.circle(frame, pt, 4, (56, 189, 248), -1)

        cv2.putText(frame, f"Max: {max(ctrl_vals):.1f}  Now: {ctrl_vals[-1]:.1f}", (px0 + 14, py0 + ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Waiting for first step...", (px0 + 75, py0 + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 155, 180), 1, cv2.LINE_AA)

    # Right panel: Per-building Storage SoC bars
    rx0, ry0, rw, rh = 398, 66, 222, 264
    cv2.rectangle(frame, (rx0, ry0), (rx0 + rw, ry0 + rh), (26, 33, 50), -1)
    cv2.rectangle(frame, (rx0, ry0), (rx0 + rw, ry0 + rh), (55, 70, 98), 1)
    cv2.putText(frame, "Building Storage SoC", (rx0 + 14, ry0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 245), 1, cv2.LINE_AA)

    socs = _read_building_socs(base_env)
    max_bars = min(len(socs), 8)
    if max_bars > 0:
        bar_area_top = ry0 + 44
        bar_step = max(24, (rh - 56) // max_bars)
        bar_h = min(14, bar_step - 10)
        bar_w = rw - 110
        for i in range(max_bars):
            b_name, soc = socs[i]
            label = b_name if len(b_name) <= 10 else b_name[:10]
            y_top = bar_area_top + i * bar_step
            cv2.putText(frame, label, (rx0 + 10, y_top + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 208, 228), 1, cv2.LINE_AA)
            bx = rx0 + 96
            cv2.rectangle(frame, (bx, y_top), (bx + bar_w, y_top + bar_h), (42, 52, 74), -1)
            fill_w = int(round(soc * bar_w))
            if fill_w > 0:
                color = (74, 222, 128) if soc >= 0.25 else (250, 204, 21)
                cv2.rectangle(frame, (bx, y_top), (bx + fill_w, y_top + bar_h), color, -1)
            cv2.putText(frame, f"{int(round(soc * 100))}%", (bx + bar_w - 30, y_top + bar_h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 245, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No storage units", (rx0 + 40, ry0 + rh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 155, 180), 1, cv2.LINE_AA)

    return frame


class CityLearn_Env(RawEnvironment):
    """Single-agent centralized wrapper for CityLearn (``central_agent=True``)."""

    def __init__(self, config: Any):
        super().__init__()
        self.env, self.base_env, self.env_id = _create_citylearn_env(config, central_agent=True)
        self.render_mode = getattr(config, "render_mode", "rgb_array")
        self.evaluate_on_done = bool(getattr(config, "evaluate_on_done", False))
        self._initial_seed = getattr(config, "env_seed", None)

        obs_spaces, self._obs_is_list = _unwrap_space_list(self.env.observation_space)
        act_spaces, self._act_is_list = _unwrap_space_list(self.env.action_space)
        self.observation_space = _to_gymnasium_box(obs_spaces[0])

        raw_act_box = _to_gymnasium_box(act_spaces[0])
        self._raw_act_low = raw_act_box.low.copy()
        self._raw_act_high = raw_act_box.high.copy()
        # Expose symmetric [-1, 1] continuous bounds for XuanCe policies.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=raw_act_box.shape, dtype=np.float32)

        self.max_episode_steps = _resolve_max_episode_steps(self.base_env, config)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._net_load_history: List[float] = []

    def evaluate_kpis(self) -> Dict[str, float]:
        """Compute district-level CityLearn KPIs for the current episode."""
        return _extract_district_kpis(self.base_env)

    def reset(self, **kwargs):
        """Reset the centralized CityLearn environment."""
        reset_kwargs = dict(kwargs)
        if "seed" not in reset_kwargs and self._initial_seed is not None:
            reset_kwargs["seed"] = int(self._initial_seed)
            self._initial_seed = None
        try:
            raw_obs, raw_info = self.env.reset(**reset_kwargs)
        except TypeError:
            raw_obs, raw_info = self.env.reset()

        obs_item = raw_obs[0] if self._obs_is_list else raw_obs
        obs_vec = _format_obs_vector(obs_item, self.observation_space.shape[0])
        self._episode_step = 0
        self._episode_reward = 0.0
        self._net_load_history = []
        init_load = _read_current_net_load(self.base_env)
        if init_load is not None:
            self._net_load_history.append(init_load)
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        return obs_vec, info

    def step(self, action):
        """Rescale ``[-1, 1]`` action to CityLearn bounds and step one hour/interval."""
        phys_act = _rescale_from_unit_box(action, self._raw_act_low, self._raw_act_high)
        env_action = [phys_act] if self._act_is_list else phys_act
        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = self.env.step(env_action)

        obs_item = raw_obs[0] if self._obs_is_list else raw_obs
        obs_vec = _format_obs_vector(obs_item, self.observation_space.shape[0])
        if isinstance(raw_rew, (list, tuple, np.ndarray)) and np.ndim(raw_rew) > 0:
            reward = float(raw_rew[0])
        else:
            reward = float(raw_rew)

        self._episode_step += 1
        self._episode_reward += reward
        curr_load = _read_current_net_load(self.base_env)
        if curr_load is not None:
            self._net_load_history.append(curr_load)

        terminated = bool(raw_term)
        truncated = bool(raw_trunc or (not terminated and self._episode_step >= self.max_episode_steps))
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        if (terminated or truncated) and self.evaluate_on_done:
            info["kpis"] = self.evaluate_kpis()

        return obs_vec, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        """Render a headless RGB dashboard frame summarizing grid load and storage SoC."""
        return _render_citylearn_dashboard(
            self.base_env,
            episode_step=self._episode_step,
            max_episode_steps=self.max_episode_steps,
            episode_reward=self._episode_reward,
            net_load_history=self._net_load_history,
            title="XuanCe x CityLearn (DRL)",
        )

    def close(self):
        """Close the CityLearn environment."""
        self.env.close()
