import argparse
import inspect
import os
from copy import deepcopy
from typing import Dict, Optional

import imageio
import numpy as np
import torch

from xuance import get_runner
from xuance.environment import make_envs


MARL_ALGOS = ("mappo", "ippo", "masac", "maddpg", "matd3")
DRL_ALGOS = ("ppo", "sac", "td3", "ddpg")
SUPPORTED_ALGOS = MARL_ALGOS + DRL_ALGOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="XuanCe x CityLearn: Smart-Grid Building Energy Coordination Demo (DRL & MARL)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=SUPPORTED_ALGOS,
        help="DRL or MARL algorithm to run (default: mappo).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="citylearn_challenge_2022_phase_1",
        help=(
            "CityLearn dataset name or local schema.json path, e.g. "
            "citylearn_challenge_2022_phase_1, citylearn_challenge_2022_phase_all, "
            "citylearn_challenge_2023_phase_1, baeda_3dem."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=("train", "test", "benchmark"),
        help="Runner execution mode (default: benchmark).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Neural network training device.",
    )
    parser.add_argument("--parallels", type=int, default=4, help="Number of parallel CityLearn environments.")
    parser.add_argument("--buffer-size", type=int, default=None, help="Optional rollout/replay buffer override.")
    parser.add_argument(
        "--episode-time-steps",
        type=int,
        default=168,
        help="Simulation timestamps per episode (e.g., 24=1 day, 168=1 week, 8760=1 year).",
    )
    parser.add_argument("--running-steps", type=int, default=16000, help="Total environment steps to train.")
    parser.add_argument("--eval-interval", type=int, default=4000, help="Evaluation interval during benchmark.")
    parser.add_argument("--test-episode", type=int, default=2, help="Number of episodes per evaluation.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--parameter-sharing",
        type=str,
        default="auto",
        choices=("auto", "true", "false"),
        help="MARL parameter sharing override ('auto' keeps the YAML default).",
    )
    parser.add_argument(
        "--pad-spaces",
        type=str,
        default="auto",
        choices=("auto", "true", "false"),
        help="MARL heterogeneous observation/action padding override ('auto' follows parameter sharing).",
    )
    parser.add_argument(
        "--eval-kpis",
        action="store_true",
        help="Run a deterministic evaluation rollout after training and print district KPIs.",
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default=None,
        help="Optional path to save a deterministic evaluation dashboard GIF.",
    )
    parser.add_argument(
        "--export-ui-dir",
        type=str,
        default=None,
        help="Optional directory to export a deterministic evaluation session for CityLearn UI (CSV telemetry + KPIs).",
    )
    parser.add_argument(
        "--ui-session-name",
        type=str,
        default=None,
        help="Optional session subfolder name inside --export-ui-dir (defaults to '<algo>_<dataset>').",
    )
    return parser.parse_args()


def _parse_tri_bool(flag_value: str) -> Optional[bool]:
    if flag_value == "true":
        return True
    if flag_value == "false":
        return False
    return None


def build_parser_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Build clean override Namespace without injecting None keys into XuanCe configs."""
    overrides = argparse.Namespace(
        device=args.device,
        parallels=args.parallels,
        episode_time_steps=args.episode_time_steps,
        running_steps=args.running_steps,
        eval_interval=args.eval_interval,
        test_episode=args.test_episode,
        seed=args.seed,
        env_seed=args.seed,
        render=False,
    )

    is_marl = args.algo in MARL_ALGOS
    if is_marl:
        ps_override = _parse_tri_bool(args.parameter_sharing)
        if ps_override is not None:
            overrides.use_parameter_sharing = ps_override
        pad_override = _parse_tri_bool(args.pad_spaces)
        if pad_override is not None:
            overrides.pad_spaces = pad_override

        if args.buffer_size is not None:
            overrides.buffer_size = args.buffer_size
        elif args.algo in ("mappo", "ippo"):
            step_budget = max(args.parallels * 4, min(672, args.running_steps))
            if step_budget % args.parallels != 0:
                step_budget = max(args.parallels, (step_budget // args.parallels) * args.parallels)
            overrides.buffer_size = step_budget
    else:
        if args.buffer_size is not None:
            if args.algo == "ppo":
                overrides.horizon_size = max(1, args.buffer_size // args.parallels)
            else:
                overrides.buffer_size = args.buffer_size
        elif args.algo == "ppo":
            target_horizon = min(168, max(8, args.running_steps // max(1, args.parallels * 2)))
            overrides.horizon_size = target_horizon

    return overrides


def _select_eval_actions(agent, obs, avail_actions, is_marl: bool):
    """Call single-agent or multi-agent ``get_actions`` respecting each algorithm's signature."""
    sig_params = inspect.signature(agent.get_actions).parameters
    if is_marl:
        kwargs = {"obs_list": obs}
        if "avail_actions_list" in sig_params:
            kwargs["avail_actions_list"] = avail_actions
        if "test_mode" in sig_params:
            kwargs["test_mode"] = True
        if "deterministic" in sig_params:
            kwargs["deterministic"] = True
        return agent.get_actions(**kwargs).env_actions

    if hasattr(agent, "_process_observation") and getattr(agent, "use_obsnorm", False):
        if hasattr(agent, "obs_rms"):
            agent.obs_rms.update(obs)
        obs = agent._process_observation(obs)

    kwargs = {}
    if "observations" in sig_params:
        kwargs["observations"] = obs
    else:
        first_param = next(iter(sig_params.keys()))
        kwargs[first_param] = obs
    if "test_mode" in sig_params:
        kwargs["test_mode"] = True
    if "deterministic" in sig_params:
        kwargs["deterministic"] = True
    return agent.get_actions(**kwargs).env_actions


def evaluate_and_render(
    runner,
    is_marl: bool,
    print_kpis: bool = True,
    gif_path: Optional[str] = None,
    export_ui_dir: Optional[str] = None,
    ui_session_name: Optional[str] = None,
    fps: int = 10,
) -> Dict[str, float]:
    """Run one deterministic evaluation episode, optionally reporting KPIs, exporting a GIF, or saving CityLearn UI files."""
    eval_config = deepcopy(runner.config)
    eval_config.parallels = 1
    eval_config.vectorize = "DummyVecMultiAgentEnv" if is_marl else "DummyVecEnv"
    eval_config.evaluate_on_done = True
    eval_config.render = bool(gif_path)
    eval_config.render_mode = "rgb_array"

    resolved_session_name: Optional[str] = None
    if export_ui_dir:
        ui_root = os.path.abspath(export_ui_dir)
        os.makedirs(ui_root, exist_ok=True)
        resolved_session_name = ui_session_name or f"{runner.config.agent.lower()}_{runner.config.env_id}"
        citylearn_kwargs = dict(getattr(eval_config, "citylearn_kwargs", None) or {})
        citylearn_kwargs["render_directory"] = ui_root
        citylearn_kwargs["render_session_name"] = resolved_session_name
        eval_config.citylearn_render_mode = "end"
        eval_config.citylearn_kwargs = citylearn_kwargs

    eval_envs = make_envs(eval_config)
    frames = []
    episode_reward = 0.0
    kpis: Dict[str, float] = {}
    ui_export_path: Optional[str] = None
    try:
        obs, _ = eval_envs.reset()
        avail_actions = getattr(eval_envs, "buf_avail_actions", None)
        if not getattr(runner.agent, "use_actions_mask", False):
            avail_actions = None

        if gif_path:
            frame = eval_envs.render("rgb_array")[0]
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

        done = False
        while not done:
            actions = _select_eval_actions(runner.agent, obs, avail_actions, is_marl=is_marl)
            obs, rewards, terminated, truncated, infos = eval_envs.step(actions)
            if is_marl:
                avail_actions = eval_envs.buf_avail_actions if getattr(runner.agent, "use_actions_mask", False) else None
                episode_reward += float(np.mean([rewards[0][k] for k in runner.agent.agent_keys]))
                done = all(terminated[0].values()) or bool(truncated[0])
            else:
                episode_reward += float(rewards[0])
                done = bool(terminated[0]) or bool(truncated[0])

            if done and isinstance(infos[0], dict) and "kpis" in infos[0]:
                kpis = dict(infos[0]["kpis"])

            if gif_path and not done:
                frame = eval_envs.render("rgb_array")[0]
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

        if export_ui_dir and hasattr(eval_envs, "envs") and eval_envs.envs:
            raw_wrapper = eval_envs.envs[0].unwrapped
            base_env = getattr(raw_wrapper, "base_env", None)
            new_folder = getattr(base_env, "new_folder_path", None)
            if new_folder:
                ui_export_path = str(new_folder)
    finally:
        eval_envs.close()

    if print_kpis:
        print(f"\n=== CityLearn District Evaluation (return={episode_reward:.3f}) ===")
        if kpis:
            for k, v in sorted(kpis.items()):
                print(f"  {k:<46s} : {v:.4f}")
        else:
            print("  No district KPIs returned by environment.")

    if export_ui_dir:
        target_dir = ui_export_path or os.path.join(os.path.abspath(export_ui_dir), resolved_session_name or "")
        file_count = len(os.listdir(target_dir)) if os.path.isdir(target_dir) else 0
        print(
            f"Exported CityLearn UI telemetry ({file_count} files) to: {target_dir}\n"
            "Import this folder in CityLearn UI (https://github.com/Soft-CPS-Research-Group/citylearn-ui)."
        )

    if gif_path:
        if frames:
            out_dir = os.path.dirname(os.path.abspath(gif_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"Saved rollout GIF ({len(frames)} frames) to: {gif_path}")
        else:
            print("Warning: CityLearn render('rgb_array') produced no frames; skipping GIF export.")

    return kpis


if __name__ == "__main__":
    cli_args = parse_args()
    runner = get_runner(
        algo=cli_args.algo,
        env="citylearn",
        env_id=cli_args.dataset,
        parser_args=build_parser_overrides(cli_args),
    )
    runner.run(mode=cli_args.mode)

    if cli_args.eval_kpis or cli_args.save_gif or cli_args.export_ui_dir:
        evaluate_and_render(
            runner,
            is_marl=(cli_args.algo in MARL_ALGOS),
            print_kpis=cli_args.eval_kpis,
            gif_path=cli_args.save_gif,
            export_ui_dir=cli_args.export_ui_dir,
            ui_session_name=cli_args.ui_session_name,
        )
