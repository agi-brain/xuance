import contextlib
import importlib.util
import os
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium
import numpy as np
import pandas as pd

from xuance import get_runner
from xuance.environment import DummyVecEnv, DummyVecMultiAgentEnv, make_envs
import xuance.environment.multi_agent_env.citylearn as citylearn_ma_mod
import xuance.environment.single_agent_env.citylearn as citylearn_sa_mod


@contextlib.contextmanager
def _temp_workdir():
    """Run a test block in an isolated temp working directory so logs/models stay clean."""
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(orig_cwd)


class _FakeCityLearnEnv:
    """Deterministic stub mimicking ``citylearn.citylearn.CityLearnEnv`` for unit tests."""

    _is_stub = True

    def __init__(
        self,
        schema,
        central_agent=True,
        episode_time_steps=4,
        random_seed=None,
        render_mode="none",
        heterogeneous=False,
        **kwargs,
    ):
        self.schema = schema
        self.central_agent = central_agent
        self.time_steps = int(episode_time_steps) if episode_time_steps is not None else 4
        self.random_seed = random_seed
        self.render_mode = render_mode
        self.extra_kwargs = kwargs
        self.time_step = 0
        self.closed = False
        self.last_actions = None

        if central_agent:
            self.buildings = [
                SimpleNamespace(
                    name="Building_1",
                    electrical_storage=SimpleNamespace(soc=np.linspace(0.2, 0.8, self.time_steps)),
                ),
                SimpleNamespace(
                    name="Building_2",
                    electrical_storage=SimpleNamespace(soc=np.linspace(0.4, 0.6, self.time_steps)),
                ),
            ]
            self.observation_space = [
                gymnasium.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            ]
            self.action_space = [
                gymnasium.spaces.Box(
                    low=np.array([0.0, -0.5], dtype=np.float32),
                    high=np.array([1.0, 0.5], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                )
            ]
        else:
            if heterogeneous:
                self.buildings = [
                    SimpleNamespace(
                        name="Building_A",
                        electrical_storage=SimpleNamespace(soc=np.linspace(0.1, 0.9, self.time_steps)),
                    ),
                    SimpleNamespace(
                        name="Building_B",
                        electrical_storage=SimpleNamespace(soc=np.linspace(0.3, 0.7, self.time_steps)),
                    ),
                ]
                self.observation_space = [
                    gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                    gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
                ]
                self.action_space = [
                    gymnasium.spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([1.0], dtype=np.float32),
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    gymnasium.spaces.Box(
                        low=np.array([-2.0, -2.0], dtype=np.float32),
                        high=np.array([2.0, 2.0], dtype=np.float32),
                        shape=(2,),
                        dtype=np.float32,
                    ),
                ]
            else:
                self.buildings = [
                    SimpleNamespace(
                        name="Building_1",
                        electrical_storage=SimpleNamespace(soc=np.linspace(0.2, 0.8, self.time_steps)),
                    ),
                    SimpleNamespace(
                        name="Building_2",
                        electrical_storage=SimpleNamespace(soc=np.linspace(0.4, 0.6, self.time_steps)),
                    ),
                ]
                self.observation_space = [
                    gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                    gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                ]
                self.action_space = [
                    gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                ]

        self.net_electricity_consumption = [1.0]
        self.net_electricity_consumption_without_storage = np.linspace(1.0, 3.0, self.time_steps)

    def _current_obs(self):
        offset = float(self.time_step) * 0.1
        return [
            [offset + float(j) * 0.01 for j in range(space.shape[0])]
            for space in self.observation_space
        ]

    def reset(self, seed=None, options=None):
        self.time_step = 0
        self.net_electricity_consumption = [1.0]
        return self._current_obs(), {}

    def step(self, actions):
        self.last_actions = [np.asarray(a, dtype=np.float32).copy() for a in actions]
        self.time_step += 1
        self.net_electricity_consumption.append(1.0 + 0.5 * self.time_step)
        terminated = self.time_step >= (self.time_steps - 1)
        rewards = [-1.0 for _ in self.action_space]
        return self._current_obs(), rewards, terminated, False, {}

    def evaluate(self):
        return pd.DataFrame(
            [
                {"cost_function": "cost_total", "value": 0.91, "name": "District", "level": "district"},
                {"cost_function": "carbon_emissions_total", "value": 0.88, "name": "District", "level": "district"},
                {"cost_function": "cost_total", "value": 0.95, "name": "Building_1", "level": "building"},
            ]
        )

    def close(self):
        self.closed = True


def _drl_stub_config(**overrides):
    defaults = dict(
        env_name="CityLearn",
        env_id="stub_dataset",
        env_seed=7,
        parallels=1,
        vectorize="DummyVecEnv",
        episode_time_steps=4,
        normalize_obs=True,
        evaluate_on_done=False,
        render=False,
        render_mode="rgb_array",
        citylearn_render_mode="none",
        distributed_training=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _marl_stub_config(**overrides):
    defaults = dict(
        env_name="citylearn",
        env_id="stub_dataset",
        env_seed=7,
        parallels=1,
        vectorize="DummyVecMultiAgentEnv",
        episode_time_steps=4,
        normalize_obs=True,
        use_parameter_sharing=False,
        pad_spaces=None,
        evaluate_on_done=False,
        render=False,
        render_mode="rgb_array",
        citylearn_render_mode="none",
        distributed_training=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestCityLearnContract(unittest.TestCase):
    def test_missing_citylearn_raises_clear_importerror(self):
        with patch.object(citylearn_sa_mod, "citylearn", None), patch.object(citylearn_sa_mod, "CityLearnEnv", None):
            with self.assertRaises(ImportError) as ctx_drl:
                make_envs(_drl_stub_config())
            self.assertIn("pip install citylearn", str(ctx_drl.exception))

            with self.assertRaises(ImportError) as ctx_marl:
                make_envs(_marl_stub_config())
            self.assertIn("pip install citylearn", str(ctx_marl.exception))

    def test_single_agent_action_rescaling_render_insulation_and_kpis(self):
        created = []

        def _factory(*args, **kwargs):
            env = _FakeCityLearnEnv(*args, **kwargs)
            created.append(env)
            return env

        with patch.object(citylearn_sa_mod, "citylearn", object()), patch.object(
            citylearn_sa_mod, "CityLearnEnv", _factory
        ):
            envs = make_envs(_drl_stub_config(evaluate_on_done=True))
            self.assertIsInstance(envs, DummyVecEnv)
            self.assertEqual(envs.max_episode_steps, 3)
            self.assertEqual(created[0].render_mode, "none")
            self.assertTrue(created[0].central_agent)
            np.testing.assert_allclose(envs.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
            np.testing.assert_allclose(envs.action_space.high, np.array([1.0, 1.0], dtype=np.float32))

            obs, _ = envs.reset()
            self.assertEqual(obs.shape, (1, 4))

            # Step 1: [-1, 1] -> [0.0, 0.5]
            _, rew1, term1, trunc1, _ = envs.step(np.array([[-1.0, 1.0]], dtype=np.float32))
            np.testing.assert_allclose(created[0].last_actions[0], np.array([0.0, 0.5], dtype=np.float32), atol=1e-6)
            self.assertAlmostEqual(float(rew1[0]), -1.0)
            self.assertFalse(term1[0])
            self.assertFalse(trunc1[0])

            # Step 2: [0, 0] -> [0.5, 0.0]; out-of-range [2.0, -2.0] would clip to [1.0, -0.5]
            envs.step(np.array([[0.0, 0.0]], dtype=np.float32))
            np.testing.assert_allclose(created[0].last_actions[0], np.array([0.5, 0.0], dtype=np.float32), atol=1e-6)

            frames = envs.render("rgb_array")
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].shape, (360, 640, 3))
            self.assertEqual(frames[0].dtype, np.uint8)

            # Step 3: hits episode_time_steps=4 -> terminated=True, truncated=False, KPIs captured before reset
            _, _, term3, trunc3, info3 = envs.step(np.array([[2.0, -2.0]], dtype=np.float32))
            np.testing.assert_allclose(created[0].last_actions[0], np.array([1.0, -0.5], dtype=np.float32), atol=1e-6)
            self.assertTrue(term3[0])
            self.assertFalse(trunc3[0])
            self.assertIn("kpis", info3[0])
            self.assertAlmostEqual(info3[0]["kpis"]["cost_total"], 0.91)
            self.assertAlmostEqual(info3[0]["kpis"]["carbon_emissions_total"], 0.88)
            self.assertIn("reset_obs", info3[0])

            envs.close()
            self.assertTrue(created[0].closed)

    def test_single_agent_early_truncation_cap(self):
        with patch.object(citylearn_sa_mod, "citylearn", object()), patch.object(
            citylearn_sa_mod, "CityLearnEnv", _FakeCityLearnEnv
        ):
            envs = make_envs(_drl_stub_config(episode_time_steps=10, max_episode_steps=2))
            self.assertEqual(envs.max_episode_steps, 2)
            envs.reset()
            _, _, term1, trunc1, _ = envs.step(np.array([[0.0, 0.0]], dtype=np.float32))
            self.assertFalse(term1[0])
            self.assertFalse(trunc1[0])
            _, _, term2, trunc2, _ = envs.step(np.array([[0.0, 0.0]], dtype=np.float32))
            self.assertFalse(term2[0])
            self.assertTrue(trunc2[0])
            envs.close()

    def test_multi_agent_heterogeneous_spaces_and_padding(self):
        created = []

        def _factory(*args, **kwargs):
            env = _FakeCityLearnEnv(*args, heterogeneous=True, **kwargs)
            created.append(env)
            return env

        with patch.object(citylearn_sa_mod, "citylearn", object()), patch.object(
            citylearn_sa_mod, "CityLearnEnv", _factory
        ):
            # 1. Unpadded heterogeneous spaces
            envs_unpadded = make_envs(_marl_stub_config(use_parameter_sharing=False, pad_spaces=False))
            self.assertIsInstance(envs_unpadded, DummyVecMultiAgentEnv)
            self.assertEqual(envs_unpadded.agents, ["Building_A", "Building_B"])
            self.assertEqual(envs_unpadded.observation_space["Building_A"].shape, (3,))
            self.assertEqual(envs_unpadded.observation_space["Building_B"].shape, (5,))
            self.assertEqual(envs_unpadded.action_space["Building_A"].shape, (1,))
            self.assertEqual(envs_unpadded.action_space["Building_B"].shape, (2,))
            self.assertEqual(envs_unpadded.state_space.shape, (8,))
            obs_u, info_u = envs_unpadded.reset()
            self.assertIsNone(info_u[0]["avail_actions"])
            np.testing.assert_allclose(
                envs_unpadded.buf_state[0],
                np.concatenate([obs_u[0]["Building_A"], obs_u[0]["Building_B"]]),
            )
            envs_unpadded.close()

            # 2. Auto-padded spaces when use_parameter_sharing=True
            created.clear()
            envs_padded = make_envs(_marl_stub_config(use_parameter_sharing=True, pad_spaces=None, evaluate_on_done=True))
            self.assertEqual(envs_padded.observation_space["Building_A"].shape, (5,))
            self.assertEqual(envs_padded.observation_space["Building_B"].shape, (5,))
            self.assertEqual(envs_padded.action_space["Building_A"].shape, (2,))
            self.assertEqual(envs_padded.action_space["Building_B"].shape, (2,))
            self.assertEqual(envs_padded.state_space.shape, (10,))

            obs_p, _ = envs_padded.reset()
            np.testing.assert_allclose(obs_p[0]["Building_A"][3:], np.array([0.0, 0.0], dtype=np.float32))

            act_dict = {
                "Building_A": np.array([1.0, 0.9], dtype=np.float32),
                "Building_B": np.array([0.0, -1.0], dtype=np.float32),
            }
            _, rew_p, term_p, trunc_p, info_p = envs_padded.step([act_dict])
            # Building_A action sliced to 1D and rescaled from [-1, 1] -> [0, 1] => [1.0]
            np.testing.assert_allclose(created[0].last_actions[0], np.array([1.0], dtype=np.float32), atol=1e-6)
            # Building_B action rescaled from [-1, 1] -> [-2, 2] => [0.0, -2.0]
            np.testing.assert_allclose(created[0].last_actions[1], np.array([0.0, -2.0], dtype=np.float32), atol=1e-6)
            self.assertAlmostEqual(rew_p[0]["Building_A"], -1.0)
            self.assertFalse(any(term_p[0].values()))
            self.assertFalse(trunc_p[0])

            frames = envs_padded.render("rgb_array")
            self.assertEqual(frames[0].shape, (360, 640, 3))
            self.assertEqual(frames[0].dtype, np.uint8)

            # Finish remaining 2 steps to verify KPI capture at episode completion
            envs_padded.step([act_dict])
            _, _, term_end, trunc_end, info_end = envs_padded.step([act_dict])
            self.assertTrue(all(term_end[0].values()))
            self.assertFalse(trunc_end[0])
            self.assertIn("kpis", info_end[0])
            self.assertAlmostEqual(info_end[0]["kpis"]["cost_total"], 0.91)
            envs_padded.close()


@unittest.skipUnless(importlib.util.find_spec("citylearn") is not None, "citylearn is not installed")
class TestCityLearnIntegration(unittest.TestCase):
    DATASET = "citylearn_challenge_2022_phase_1"

    def test_vectorized_single_and_multi_agent_subproc_and_render(self):
        # Single-agent SubprocVecEnv
        sa_cfg = _drl_stub_config(
            env_id=self.DATASET,
            parallels=2,
            vectorize="SubprocVecEnv",
            episode_time_steps=24,
            evaluate_on_done=True,
        )
        sa_envs = make_envs(sa_cfg)
        try:
            self.assertEqual(sa_envs.max_episode_steps, 23)
            obs, _ = sa_envs.reset()
            self.assertEqual(obs.shape[0], 2)
            actions = np.stack([sa_envs.action_space.sample() for _ in range(2)], axis=0)
            next_obs, rew, term, trunc, info = sa_envs.step(actions)
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertEqual(rew.shape, (2,))
            frames = sa_envs.render("rgb_array")
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].dtype, np.uint8)
        finally:
            sa_envs.close()

        # Multi-agent SubprocVecMultiAgentEnv
        ma_cfg = _marl_stub_config(
            env_id=self.DATASET,
            parallels=2,
            vectorize="SubprocVecMultiAgentEnv",
            episode_time_steps=24,
            use_parameter_sharing=True,
            evaluate_on_done=True,
        )
        ma_envs = make_envs(ma_cfg)
        try:
            self.assertEqual(ma_envs.num_agents, 5)
            self.assertEqual(ma_envs.max_episode_steps, 23)
            obs_list, info_list = ma_envs.reset()
            self.assertEqual(len(obs_list), 2)
            self.assertEqual(len(info_list[0]["state"]), int(np.prod(ma_envs.state_space.shape)))
            actions = [{k: ma_envs.action_space[k].sample() for k in ma_envs.agents} for _ in range(2)]
            next_obs, rews, terms, truncs, infos = ma_envs.step(actions)
            self.assertEqual(len(next_obs), 2)
            self.assertEqual(len(rews[0]), 5)
            frames = ma_envs.render("rgb_array")
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].dtype, np.uint8)
        finally:
            ma_envs.close()

    def test_drl_configs_end_to_end_smoke(self):
        with _temp_workdir():
            for algo in ("ppo", "sac", "td3", "ddpg"):
                with self.subTest(algo=algo):
                    overrides = dict(
                        dl_toolbox="torch",
                        device="cpu",
                        parallels=2,
                        episode_time_steps=24,
                        running_steps=24,
                        eval_interval=12,
                        test_episode=1,
                        render=False,
                    )
                    if algo == "ppo":
                        overrides.update(horizon_size=12, n_epochs=1, n_minibatch=2)
                    else:
                        overrides.update(buffer_size=64, batch_size=8, start_training=8, training_frequency=4)

                    runner = get_runner(
                        algo=algo,
                        env="citylearn",
                        env_id=self.DATASET,
                        parser_args=Namespace(**overrides),
                    )
                    runner.run(mode="benchmark", running_steps=24, eval_interval=12, test_episodes=1)

    def test_marl_configs_end_to_end_smoke(self):
        with _temp_workdir():
            for algo in ("mappo", "ippo", "masac", "matd3", "maddpg"):
                with self.subTest(algo=algo):
                    overrides = dict(
                        dl_toolbox="torch",
                        device="cpu",
                        parallels=2,
                        episode_time_steps=24,
                        running_steps=24,
                        eval_interval=12,
                        test_episode=1,
                        render=False,
                    )
                    if algo in ("mappo", "ippo"):
                        overrides.update(buffer_size=24, n_epochs=1, n_minibatch=2)
                    else:
                        overrides.update(buffer_size=64, batch_size=8, start_training=8, training_frequency=4)

                    runner = get_runner(
                        algo=algo,
                        env="citylearn",
                        env_id=self.DATASET,
                        parser_args=Namespace(**overrides),
                    )
                    runner.run(mode="benchmark", running_steps=24, eval_interval=12, test_episodes=1)


if __name__ == "__main__":
    unittest.main()
