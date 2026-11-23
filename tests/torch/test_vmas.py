import importlib.util
from argparse import Namespace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from xuance import get_runner
from xuance.environment import NativeVecMultiAgentEnv, make_envs
import xuance.environment.multi_agent_env.vmas as vmas_mod


class _LegacyBox:
    """Minimal duck-typed legacy gym.spaces.Box stand-in."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = np.dtype(dtype)


class _LegacyDiscrete:
    """Minimal duck-typed legacy gym.spaces.Discrete stand-in."""

    def __init__(self, n):
        self.n = int(n)


class _LegacyMultiDiscrete:
    """Minimal duck-typed legacy gym.spaces.MultiDiscrete stand-in."""

    def __init__(self, nvec):
        self.nvec = np.asarray(nvec, dtype=np.int64)


class _LegacyDict:
    """Minimal duck-typed legacy gym.spaces.Dict stand-in."""

    def __init__(self, spaces_dict):
        self.spaces = dict(spaces_dict)


class _FakeVMASBatchedEnv:
    """Lightweight deterministic batched VMAS stub for CI contract tests."""

    def __init__(
        self,
        num_envs=3,
        continuous_actions=True,
        terminated_truncated=True,
        multidiscrete_actions=False,
        max_steps=4,
    ):
        self.num_envs = num_envs
        self.continuous_actions = continuous_actions
        self.terminated_truncated = terminated_truncated
        self.max_steps = max_steps
        self.agents = [SimpleNamespace(name="agent_0"), SimpleNamespace(name="agent_1")]
        self.observation_space = _LegacyDict(
            {"agent_0": _LegacyBox(-np.inf, np.inf, (3,)), "agent_1": _LegacyBox(-np.inf, np.inf, (4,))}
        )
        if multidiscrete_actions:
            act_space = _LegacyMultiDiscrete([3, 3])
        elif continuous_actions:
            act_space = _LegacyBox(-1.0, 1.0, (2,))
        else:
            act_space = _LegacyDiscrete(5)
        self.action_space = _LegacyDict({"agent_0": act_space, "agent_1": act_space})

        self.step_counts = torch.zeros(num_envs, dtype=torch.long)
        self.last_actions = None
        self.reset_at_calls = []
        self.viewer = SimpleNamespace(closed=False, close=self._close_viewer)
        self.early_terminate_env = None

    def _close_viewer(self):
        self.viewer.closed = True

    def _obs_batch(self, offset=0.0):
        obs_0 = torch.arange(self.num_envs * 3, dtype=torch.float32).reshape(self.num_envs, 3) + offset
        obs_1 = torch.arange(self.num_envs * 4, dtype=torch.float32).reshape(self.num_envs, 4) + 10.0 + offset
        return {"agent_0": obs_0, "agent_1": obs_1}

    def reset(self):
        self.step_counts.zero_()
        return self._obs_batch(offset=0.0)

    def step(self, actions):
        self.last_actions = actions
        self.step_counts += 1
        offset = float(self.step_counts.max().item())
        obs = self._obs_batch(offset=offset)
        rews = {
            "agent_0": torch.full((self.num_envs,), 1.5, dtype=torch.float32),
            "agent_1": torch.full((self.num_envs,), 0.5, dtype=torch.float32),
        }
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.early_terminate_env is not None:
            term[self.early_terminate_env] = True
            self.early_terminate_env = None
        trunc = self.step_counts >= self.max_steps
        if self.terminated_truncated:
            return obs, rews, term, trunc, {"agent_0": {}, "agent_1": {}}
        return obs, rews, torch.logical_or(term, trunc), {"agent_0": {}, "agent_1": {}}

    def reset_at(self, index):
        self.reset_at_calls.append(index)
        self.step_counts[index] = 0
        obs = self._obs_batch(offset=-1.0)
        obs["agent_0"][index] = torch.tensor([100.0 + index, 101.0, 102.0])
        obs["agent_1"][index] = torch.tensor([200.0 + index, 201.0, 202.0, 203.0])
        return obs

    def render(self, mode="rgb_array", env_index=0):
        return np.full((16, 16, 3), fill_value=env_index, dtype=np.uint8)


class _FakeVMASModule:
    def __init__(self, env_factory):
        self._env_factory = env_factory

    def make_env(
        self,
        scenario,
        num_envs,
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=None,
        dict_spaces=True,
        terminated_truncated=False,
        clamp_actions=False,
        **kwargs,
    ):
        return self._env_factory(
            scenario=scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            max_steps=max_steps,
            seed=seed,
            dict_spaces=dict_spaces,
            terminated_truncated=terminated_truncated,
            clamp_actions=clamp_actions,
            **kwargs,
        )


def _stub_config(**kwargs):
    defaults = dict(
        env_name="vmas",
        env_id="transport",
        env_seed=1,
        parallels=3,
        vectorize="NativeVecMultiAgentEnv",
        vmas_device="cpu",
        continuous_action=True,
        max_episode_steps=4,
        render_mode="rgb_array",
        scenario_kwargs={},
        distributed_training=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestVMASNativeVecContract(unittest.TestCase):
    def test_missing_vmas_raises_clear_importerror(self):
        with patch.object(vmas_mod, "vmas", None):
            with self.assertRaises(ImportError) as ctx:
                make_envs(_stub_config())
            self.assertIn("pip install vmas", str(ctx.exception))

    def test_multidiscrete_action_space_raises_not_implemented(self):
        fake_pkg = _FakeVMASModule(lambda **kw: _FakeVMASBatchedEnv(num_envs=kw["num_envs"], multidiscrete_actions=True))
        with patch.object(vmas_mod, "vmas", fake_pkg):
            with self.assertRaises(NotImplementedError):
                make_envs(_stub_config(continuous_action=False))

    def test_make_envs_contract_spaces_state_and_partial_resets_5tuple(self):
        created = {}

        def _factory(**kw):
            env = _FakeVMASBatchedEnv(
                num_envs=kw["num_envs"],
                continuous_actions=kw["continuous_actions"],
                terminated_truncated=kw["terminated_truncated"],
                max_steps=kw["max_steps"],
            )
            created["env"] = env
            return env

        with patch.object(vmas_mod, "vmas", _FakeVMASModule(_factory)):
            envs = make_envs(_stub_config(parallels=3, max_episode_steps=2))
            self.assertIsInstance(envs, NativeVecMultiAgentEnv)
            self.assertEqual(envs.num_envs, 3)
            self.assertEqual(envs.agents, ["agent_0", "agent_1"])
            self.assertEqual(envs.num_agents, 2)
            self.assertIsInstance(envs.observation_space["agent_0"], gymnasium.spaces.Box)
            self.assertIsInstance(envs.action_space["agent_0"], gymnasium.spaces.Box)
            self.assertEqual(envs.state_space.shape, (7,))

            obs_list, info = envs.reset()
            self.assertEqual(len(obs_list), 3)
            self.assertEqual(len(envs.buf_obs), 3)
            self.assertEqual(len(envs.buf_state), 3)
            np.testing.assert_allclose(
                envs.buf_state[1],
                np.concatenate([obs_list[1]["agent_0"], obs_list[1]["agent_1"]]),
            )
            self.assertIsNone(info[0]["avail_actions"])

            # Step 1: early termination only on env 0 + out-of-range action clipping check.
            created["env"].early_terminate_env = 0
            actions = [{"agent_0": np.array([2.5, -3.0]), "agent_1": np.array([0.5, -0.5])} for _ in range(3)]
            obs1, rew1, term1, trunc1, info1 = envs.step(actions)
            self.assertTrue(torch.all(created["env"].last_actions["agent_0"] <= 1.0))
            self.assertTrue(torch.all(created["env"].last_actions["agent_0"] >= -1.0))
            self.assertEqual(created["env"].reset_at_calls, [0])
            self.assertTrue(all(term1[0].values()))
            self.assertFalse(trunc1[0])
            self.assertFalse(any(term1[1].values()))
            self.assertFalse(trunc1[1])
            self.assertEqual(info1[0]["episode_step"], 1)
            self.assertAlmostEqual(info1[0]["episode_score"]["agent_0"], 1.5)
            self.assertIn("reset_obs", info1[0])
            self.assertNotIn("reset_obs", info1[1])
            np.testing.assert_allclose(info1[0]["reset_obs"]["agent_0"], np.array([100.0, 101.0, 102.0]))
            np.testing.assert_allclose(
                info1[0]["reset_state"],
                np.array([100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0]),
            )

            # Step 2: envs 1 and 2 hit max_episode_steps=2 (truncation), env 0 is at step 1 after reset.
            obs2, rew2, term2, trunc2, info2 = envs.step(actions)
            self.assertEqual(created["env"].reset_at_calls, [0, 1, 2])
            self.assertFalse(trunc2[0])
            self.assertTrue(trunc2[1])
            self.assertTrue(trunc2[2])
            self.assertFalse(any(term2[1].values()))
            self.assertEqual(info2[0]["episode_step"], 1)
            self.assertEqual(info2[1]["episode_step"], 2)
            self.assertAlmostEqual(info2[1]["episode_score"]["agent_0"], 3.0)

            frames = envs.render("rgb_array")
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[2].shape, (16, 16, 3))
            viewer = created["env"].viewer
            envs.close()
            self.assertTrue(viewer.closed)
            self.assertIsNone(created["env"].viewer)

    def test_4tuple_fallback_and_discrete_action_packing(self):
        created = {}

        def _factory(**kw):
            env = _FakeVMASBatchedEnv(
                num_envs=kw["num_envs"],
                continuous_actions=False,
                terminated_truncated=False,
                max_steps=2,
            )
            created["env"] = env
            return env

        fake_pkg = _FakeVMASModule(_factory)
        with patch.object(vmas_mod, "vmas", fake_pkg):
            # Hide terminated_truncated from inspect.signature to exercise 4-tuple branch.
            fake_pkg.make_env = lambda scenario, num_envs, device="cpu", continuous_actions=True, max_steps=None, seed=None, dict_spaces=True, **kw: _factory(
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                terminated_truncated=False,
                max_steps=max_steps,
            )
            envs = make_envs(_stub_config(parallels=2, continuous_action=False, max_episode_steps=2))
            self.assertIsInstance(envs.action_space["agent_0"], gymnasium.spaces.Discrete)
            obs_list, info = envs.reset()
            self.assertEqual(info[0]["avail_actions"]["agent_0"].dtype, np.bool_)

            created["env"].early_terminate_env = 0
            _, _, term1, trunc1, _ = envs.step([{"agent_0": 2, "agent_1": 4}, {"agent_0": 1, "agent_1": 3}])
            self.assertEqual(tuple(created["env"].last_actions["agent_0"].shape), (2, 1))
            self.assertEqual(created["env"].last_actions["agent_0"].dtype, torch.long)
            self.assertTrue(all(term1[0].values()))
            self.assertFalse(trunc1[0])

            _, _, term2, trunc2, _ = envs.step([{"agent_0": 0, "agent_1": 0}, {"agent_0": 1, "agent_1": 1}])
            self.assertFalse(trunc2[0])
            self.assertTrue(trunc2[1])
            self.assertFalse(any(term2[1].values()))
            envs.close()


@unittest.skipUnless(importlib.util.find_spec("vmas") is not None, "vmas is not installed")
class TestVMASIntegration(unittest.TestCase):
    @staticmethod
    def _base_args(**overrides):
        args = Namespace(
            dl_toolbox="torch",
            device="cpu",
            vmas_device="cpu",
            parallels=4,
            running_steps=64,
            eval_interval=32,
            max_episode_steps=16,
            test_episode=1,
            render=False,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_mappo_transport_native_vec(self):
        args = self._base_args(buffer_size=32, n_epochs=1, n_minibatch=2)
        runner = get_runner(algo="mappo", env="vmas", env_id="transport", parser_args=args)
        runner.run(mode="benchmark", running_steps=64, eval_interval=32, test_episodes=1)

    def test_masac_navigation_native_vec(self):
        args = self._base_args(buffer_size=128, batch_size=8, start_training=16, training_frequency=8)
        runner = get_runner(algo="masac", env="vmas", env_id="navigation", parser_args=args)
        runner.run(mode="benchmark", running_steps=64, eval_interval=32, test_episodes=1)

    def test_qmix_navigation_discrete_native_vec(self):
        args = self._base_args(
            continuous_action=False,
            buffer_size=128,
            batch_size=8,
            start_training=16,
            training_frequency=8,
        )
        runner = get_runner(algo="qmix", env="vmas", env_id="navigation", parser_args=args)
        runner.run(mode="benchmark", running_steps=64, eval_interval=32, test_episodes=1)

    def test_mappo_dummy_vec_fallback(self):
        args = self._base_args(
            vectorize="DummyVecMultiAgentEnv",
            parallels=2,
            buffer_size=32,
            n_epochs=1,
            n_minibatch=2,
        )
        runner = get_runner(algo="mappo", env="vmas", env_id="navigation", parser_args=args)
        runner.run(mode="benchmark", running_steps=64, eval_interval=32, test_episodes=1)

    def test_vmas_render_rgb_array(self):
        envs = make_envs(
            _stub_config(
                env_id="transport",
                parallels=2,
                max_episode_steps=10,
                render=True,
                render_mode="rgb_array",
            )
        )
        try:
            envs.reset()
            try:
                frames = envs.render("rgb_array")
            except Exception as exc:
                self.skipTest(f"Headless OpenGL rendering unavailable: {exc}")
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].ndim, 3)
            self.assertEqual(frames[0].shape[-1], 3)
            self.assertEqual(frames[0].dtype, np.uint8)
        finally:
            envs.close()


if __name__ == "__main__":
    unittest.main()
