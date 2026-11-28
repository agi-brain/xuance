import os
import tempfile
import unittest
from argparse import Namespace

import gymnasium as gym
import numpy as np
import torch

from xuance import get_runner
from xuance.torch.agents import REGISTRY_Agents, TRPO_Agent
from xuance.torch.learners import REGISTRY_Learners, TRPO_Learner
from xuance.torch.learners.policy_gradient.trpo_learner import (
    _get_flat_params,
    conjugate_gradient,
)


def _make_trpo_config(tmpdir: str,
                      discrete: bool = True,
                      sub_sampling_factor: int = 1,
                      target_kl: float = 0.01,
                      line_search_max_iter: int = 10,
                      ent_coef: float = 0.0,
                      representation: str = "Basic_MLP") -> Namespace:
    return Namespace(
        agent="TRPO",
        dl_toolbox="torch",
        device="cpu",
        distributed_training=False,
        seed=1,
        env_name="Classic Control",
        env_id="CartPole-v1" if discrete else "Pendulum-v1",
        env_seed=1,
        vectorize="DummyVecEnv",
        representation=representation,
        policy="Categorical_AC" if discrete else "Gaussian_AC",
        learner="TRPO_Learner",
        runner="DRL",
        representation_hidden_size=[32] if representation == "Basic_MLP" else None,
        actor_hidden_size=[32, 32] if representation == "Basic_Identical" else [32],
        critic_hidden_size=[32, 32] if representation == "Basic_Identical" else [32],
        activation="tanh",
        activation_action="tanh",
        parallels=2,
        running_steps=256,
        horizon_size=32,
        n_epochs=2,
        n_minibatch=2,
        learning_rate=1e-3,
        target_kl=target_kl,
        cg_max_steps=10,
        cg_damping=0.1,
        cg_residual_tol=1e-10,
        line_search_max_iter=line_search_max_iter,
        line_search_shrinking_factor=0.8,
        sub_sampling_factor=sub_sampling_factor,
        ent_coef=ent_coef,
        gamma=0.99,
        use_gae=True,
        gae_lambda=0.95,
        use_advnorm=True,
        use_grad_clip=True,
        grad_clip_norm=0.5,
        use_actions_mask=False,
        use_obsnorm=False,
        use_rewnorm=False,
        obsnorm_range=5,
        rewnorm_range=5,
        render=False,
        render_mode="rgb_array",
        fps=50,
        logger="tensorboard",
        log_dir=os.path.join(tmpdir, "logs"),
        model_dir=os.path.join(tmpdir, "models"),
        eval_interval=128,
        test_episode=2,
    )


def _make_fake_samples(obs_dim: int,
                       action_space: gym.Space,
                       batch_size: int = 64,
                       seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((batch_size, obs_dim)).astype(np.float32)
    if isinstance(action_space, gym.spaces.Discrete):
        actions = rng.integers(0, action_space.n, size=(batch_size,)).astype(np.float32)
    else:
        actions = rng.uniform(-1.0, 1.0, size=(batch_size,) + action_space.shape).astype(np.float32)
    advantages = rng.standard_normal((batch_size,)).astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = rng.standard_normal((batch_size,)).astype(np.float32)
    return {
        "obs": obs,
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
    }


class TestTRPOUnit(unittest.TestCase):
    def test_registries(self):
        self.assertIs(REGISTRY_Agents["TRPO"], TRPO_Agent)
        self.assertIs(REGISTRY_Learners["TRPO_Learner"], TRPO_Learner)

    def test_conjugate_gradient_spd_accuracy(self):
        torch.manual_seed(0)
        m = torch.randn(12, 12)
        A = m.T @ m + 0.5 * torch.eye(12)
        b = torch.randn(12)
        x_true = torch.linalg.solve(A, b)
        x_cg = conjugate_gradient(lambda v: A @ v, b, max_steps=20, residual_tol=1e-12)
        self.assertTrue(torch.allclose(x_cg, x_true, atol=1e-4, rtol=1e-4))

        # Zero right-hand side should return zero cleanly.
        x_zero = conjugate_gradient(lambda v: A @ v, torch.zeros_like(b), max_steps=10)
        self.assertTrue(torch.allclose(x_zero, torch.zeros_like(b)))

    def test_snapshot_isolation_and_multidim_gaussian_kl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
            act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            cfg = _make_trpo_config(tmpdir, discrete=False)
            agent = TRPO_Agent(config=cfg, envs=None, observation_space=obs_space, action_space=act_space)
            try:
                obs = torch.randn(16, 6)
                dist_old_ref = agent.model.actor(obs).distributions
                old_snapshot = agent.learner._snapshot_distribution(dist_old_ref)
                old_loc_copy = old_snapshot.loc.clone()

                # Perturb actor parameters and run a second forward pass (which mutates policy_distribution in place).
                with torch.no_grad():
                    for p in agent.learner.actor_params:
                        p.add_(0.1)
                dist_new = agent.model.actor(obs).distributions

                # Snapshot must remain unchanged despite actor head reusing policy_distribution.
                self.assertTrue(torch.allclose(old_snapshot.loc, old_loc_copy))

                # Multi-dimensional Gaussian KL must sum across action dims (4), not average across them.
                kl_computed = agent.learner._compute_kl(old_snapshot, dist_new)
                kl_manual = torch.distributions.kl_divergence(old_snapshot, dist_new.distribution).sum(dim=-1).mean()
                self.assertAlmostEqual(kl_computed.item(), kl_manual.item(), places=6)
                self.assertGreater(kl_computed.item(), 0.0)
            finally:
                agent.finish()

    def test_fvp_symmetry_and_psd_discrete_and_continuous(self):
        for discrete, act_space, sub_factor in [
            (True, gym.spaces.Discrete(3), 1),
            (False, gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32), 1),
            (False, gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32), 2),
        ]:
            with tempfile.TemporaryDirectory() as tmpdir:
                obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
                cfg = _make_trpo_config(tmpdir, discrete=discrete, sub_sampling_factor=sub_factor)
                agent = TRPO_Agent(config=cfg, envs=None, observation_space=obs_space, action_space=act_space)
                try:
                    obs = torch.randn(32, 5)
                    with torch.no_grad():
                        old_dist = agent.model.actor(obs).distributions
                        obs_fvp = obs[::sub_factor]
                        old_snap = agent.learner._snapshot_distribution(old_dist, step=sub_factor)

                    n_params = _get_flat_params(agent.learner.actor_params).numel()
                    torch.manual_seed(7)
                    u = torch.randn(n_params)
                    v = torch.randn(n_params)

                    Fu = agent.learner._fisher_vector_product(obs_fvp, old_snap, u)
                    Fv = agent.learner._fisher_vector_product(obs_fvp, old_snap, v)

                    # Positive definite with damping: v^T (F + damping I) v >= damping * ||v||^2.
                    vFv = torch.dot(v, Fv).item()
                    self.assertGreaterEqual(vFv, cfg.cg_damping * torch.dot(v, v).item() - 1e-5)

                    # Symmetry: u^T F v == v^T F u.
                    uFv = torch.dot(u, Fv).item()
                    vFu = torch.dot(v, Fu).item()
                    self.assertAlmostEqual(uFv, vFu, delta=1e-4 * max(1.0, abs(uFv)))
                finally:
                    agent.finish()

    def test_trust_region_constraint_parameter_separation_and_rollback(self):
        for discrete, act_space in [
            (True, gym.spaces.Discrete(2)),
            (False, gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)),
        ]:
            with tempfile.TemporaryDirectory() as tmpdir:
                obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                cfg = _make_trpo_config(tmpdir, discrete=discrete, target_kl=0.01, ent_coef=0.01)
                agent = TRPO_Agent(config=cfg, envs=None, observation_space=obs_space, action_space=act_space)
                try:
                    samples = _make_fake_samples(obs_dim=4, action_space=act_space, batch_size=64)
                    critic_before = _get_flat_params(list(agent.model.critic.parameters())).clone()
                    actor_before = _get_flat_params(agent.learner.actor_params).clone()

                    # Actor update must obey KL <= target_kl and never touch critic weights.
                    actor_info = agent.learner.update_actor(**samples)
                    self.assertLessEqual(actor_info["kl"], cfg.target_kl + 1e-7)
                    self.assertGreaterEqual(actor_info["kl"], 0.0)
                    self.assertEqual(actor_info["line_search_success"], 1.0)
                    critic_after_actor = _get_flat_params(list(agent.model.critic.parameters()))
                    self.assertTrue(torch.equal(critic_before, critic_after_actor))
                    actor_after_actor = _get_flat_params(agent.learner.actor_params).clone()
                    self.assertFalse(torch.equal(actor_before, actor_after_actor))

                    # Critic update must touch critic weights and never touch actor weights.
                    critic_info = agent.learner.update_critic(**samples)
                    self.assertIn("critic-loss", critic_info)
                    self.assertTrue(torch.equal(actor_after_actor, _get_flat_params(agent.learner.actor_params)))
                    self.assertFalse(torch.equal(critic_before, _get_flat_params(list(agent.model.critic.parameters()))))

                    # Combined learner.update() should work cleanly too.
                    merged_info = agent.learner.update(**samples)
                    self.assertIn("actor-loss", merged_info)
                    self.assertIn("critic-loss", merged_info)

                    # Line-search failure (max_iter=0) must restore actor params exactly.
                    agent.learner.line_search_max_iter = 0
                    actor_pre_fail = _get_flat_params(agent.learner.actor_params).clone()
                    fail_info = agent.learner.update_actor(**samples)
                    self.assertEqual(fail_info["line_search_success"], 0.0)
                    self.assertEqual(fail_info["kl"], 0.0)
                    self.assertTrue(torch.equal(actor_pre_fail, _get_flat_params(agent.learner.actor_params)))
                finally:
                    agent.finish()

    def test_save_and_load_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            act_space = gym.spaces.Discrete(2)
            cfg = _make_trpo_config(tmpdir, discrete=True)
            cfg.use_obsnorm = True
            agent_a = TRPO_Agent(config=cfg, envs=None, observation_space=obs_space, action_space=act_space)
            agent_b = TRPO_Agent(config=cfg, envs=None, observation_space=obs_space, action_space=act_space)
            try:
                samples = _make_fake_samples(obs_dim=4, action_space=act_space, batch_size=64)
                agent_a.obs_rms.update(samples["obs"])
                agent_a.learner.update(**samples)
                save_dir = os.path.join(tmpdir, "ckpt")
                agent_a.save_model(model_name="trpo_ckpt", model_path=save_dir)

                agent_b.load_model(path=save_dir, model="trpo_ckpt")
                for pa, pb in zip(agent_a.model.parameters(), agent_b.model.parameters()):
                    self.assertTrue(torch.allclose(pa, pb))
                self.assertTrue(np.allclose(agent_a.obs_rms.mean, agent_b.obs_rms.mean))
            finally:
                agent_a.finish()
                agent_b.finish()

    def test_runner_smoke_discrete_and_continuous(self):
        for env_id in ["CartPole-v1", "Pendulum-v1"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                parser_args = Namespace(
                    dl_toolbox="torch",
                    device="cpu",
                    parallels=2,
                    horizon_size=32,
                    running_steps=128,
                    eval_interval=64,
                    test_episode=2,
                    n_epochs=2,
                    n_minibatch=2,
                    render=False,
                    log_dir=os.path.join(tmpdir, "logs"),
                    model_dir=os.path.join(tmpdir, "models"),
                )
                runner = get_runner(algo="trpo", env="classic_control", env_id=env_id, parser_args=parser_args)
                runner.run(mode="train", running_steps=128)


if __name__ == "__main__":
    unittest.main()
