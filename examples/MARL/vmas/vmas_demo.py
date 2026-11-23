import argparse
import os
from copy import deepcopy

import imageio
import numpy as np
import torch

from xuance import get_runner
from xuance.environment import make_envs


SUPPORTED_ALGOS = ("mappo", "ippo", "masac", "maddpg", "matd3", "qmix")


def parse_args():
    parser = argparse.ArgumentParser(
        description="XuanCe x VMAS: Native Vectorized Multi-Robot Physics Simulation Demo"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=SUPPORTED_ALGOS,
        help="MARL algorithm to run (default: mappo).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="transport",
        help=(
            "VMAS scenario name, e.g. transport, balance, navigation, wheel, "
            "give_way, flocking, dispersion, football."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=("train", "test", "benchmark"),
        help="Runner execution mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Neural network training device.",
    )
    parser.add_argument(
        "--vmas-device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="VMAS physics simulation device ('cpu' or 'cuda').",
    )
    parser.add_argument("--parallels", type=int, default=32, help="Number of batched VMAS environments.")
    parser.add_argument("--buffer-size", type=int, default=None, help="Optional buffer size override.")
    parser.add_argument("--running-steps", type=int, default=200000, help="Total environment steps to train.")
    parser.add_argument("--eval-interval", type=int, default=50000, help="Evaluation interval during benchmark.")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Horizon per episode (VMAS max_steps).")
    parser.add_argument("--test-episode", type=int, default=5, help="Number of episodes per evaluation.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Optional override for number of agents (forwarded via scenario_kwargs).",
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default=None,
        help="Optional path to save a greedy evaluation rollout GIF after running.",
    )
    return parser.parse_args()


def build_parser_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Build clean override Namespace without injecting None keys into XuanCe configs."""
    overrides = argparse.Namespace(
        device=args.device,
        vmas_device=args.vmas_device,
        parallels=args.parallels,
        running_steps=args.running_steps,
        eval_interval=args.eval_interval,
        max_episode_steps=args.max_episode_steps,
        test_episode=args.test_episode,
        seed=args.seed,
        env_seed=args.seed,
        render=False,
    )
    if args.n_agents is not None:
        overrides.scenario_kwargs = {"n_agents": args.n_agents}

    # Ensure on-policy buffer_size remains evenly divisible by parallels.
    if args.buffer_size is not None:
        overrides.buffer_size = args.buffer_size
    elif args.algo in ("mappo", "ippo"):
        default_buffer_size = 3200
        if default_buffer_size % args.parallels != 0:
            overrides.buffer_size = max(args.parallels, (default_buffer_size // args.parallels) * args.parallels)
    return overrides


def save_rollout_gif(runner, gif_path: str, fps: int = 20) -> float:
    """Run one deterministic rollout episode and export an RGB-array GIF."""
    gif_config = deepcopy(runner.config)
    gif_config.parallels = 1
    gif_config.render = True
    gif_config.render_mode = "rgb_array"

    gif_envs = make_envs(gif_config)
    frames = []
    episode_reward = 0.0
    try:
        obs_list, _ = gif_envs.reset()
        avail_actions = gif_envs.buf_avail_actions if getattr(runner.agent, "use_actions_mask", False) else None
        frame = gif_envs.render("rgb_array")[0]
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        done = False
        while not done:
            policy_out = runner.agent.get_actions(
                obs_list=obs_list,
                avail_actions_list=avail_actions,
                test_mode=True,
                deterministic=True,
            )
            obs_list, rewards_list, terminated_list, truncated, info = gif_envs.step(policy_out.env_actions)
            avail_actions = gif_envs.buf_avail_actions if getattr(runner.agent, "use_actions_mask", False) else None
            episode_reward += float(np.mean([rewards_list[0][k] for k in runner.agent.agent_keys]))
            done = all(terminated_list[0].values()) or bool(truncated[0])
            if not done:
                frame = gif_envs.render("rgb_array")[0]
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        gif_envs.close()

    if frames:
        out_dir = os.path.dirname(os.path.abspath(gif_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved rollout GIF ({len(frames)} frames, mean agent return={episode_reward:.3f}) to: {gif_path}")
    else:
        print("Warning: VMAS render('rgb_array') produced no frames; skipping GIF export.")
    return episode_reward


if __name__ == "__main__":
    cli_args = parse_args()
    runner = get_runner(
        algo=cli_args.algo,
        env="vmas",
        env_id=cli_args.scenario,
        parser_args=build_parser_overrides(cli_args),
    )
    runner.run(mode=cli_args.mode)

    if cli_args.save_gif:
        save_rollout_gif(runner, cli_args.save_gif)
