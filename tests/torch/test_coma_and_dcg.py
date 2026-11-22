# Regression tests for COMA HuberLoss initialization and pure-PyTorch DCG scatter_add.

from argparse import Namespace
from types import SimpleNamespace
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from xuance import get_runner
from xuance.common import AgentGrouping, MultiAgentBaseCallback
from xuance.torch.learners.base.marl_learner import OnPolicyMultiAgentLearner
from xuance.torch.learners.multi_agent_rl.coma_learner import COMA_Learner
from xuance.torch.rl_models.heads.coordination_graph import Coordination_Graph, scatter_add


class _DummyOnPolicyMARLLearner(OnPolicyMultiAgentLearner):
    def build_optimizer(self):
        self.optimizer = None

    def update(self, *args, **kwargs):
        return {}


class _DummyCOMAModel(torch.nn.Module):
    def __init__(self, agent_keys, n_actions=5):
        super().__init__()
        self.actors = torch.nn.Linear(4, n_actions)
        self.critics = torch.nn.Linear(4, n_actions)
        self.critics.action_space = {k: gym.spaces.Discrete(n_actions) for k in agent_keys}


def _make_learner_config(log_dir: str, use_huber_loss: bool, huber_delta):
    return SimpleNamespace(
        device="cpu",
        distributed_training=False,
        log_dir=log_dir,
        model_dir=log_dir,
        episode_length=25,
        parallels=2,
        use_parameter_sharing=True,
        gamma=0.99,
        use_grad_clip=False,
        grad_clip_norm=10.0,
        use_actions_mask=False,
        use_linear_lr_decay=False,
        end_factor_lr_decay=0.5,
        learning_rate=1e-3,
        learning_rate_actor=1e-3,
        learning_rate_critic=1e-3,
        use_rnn=False,
        running_steps=64,
        buffer_size=32,
        n_epochs=1,
        n_minibatch=1,
        sync_frequency=10,
        use_value_clip=False,
        value_clip_range=None,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        use_value_norm=False,
        vf_coef=0.1,
        ent_coef=0.01,
    )


def _make_marl_smoke_args(tmpdir: str, **overrides) -> Namespace:
    params = dict(
        dl_toolbox="torch",
        device="cpu",
        vectorize="DummyVecMultiAgentEnv",
        parallels=2,
        running_steps=60,
        eval_interval=30,
        test_episode=2,
        start_training=20,
        buffer_size=40,
        batch_size=4,
        n_epochs=1,
        n_minibatch=1,
        logger="tensorboard",
        log_dir=tmpdir,
        model_dir=tmpdir,
        render=False,
    )
    params.update(overrides)
    return Namespace(**params)


class TestCOMAAndDCGFixes(unittest.TestCase):
    """Verify COMA HuberLoss(delta=None) handling and pure-PyTorch DCG graph ops."""

    def test_coma_and_onpolicy_marl_learner_huber_loss_init(self):
        agent_keys = ["agent_0", "agent_1", "agent_2"]
        grouping = AgentGrouping.shared(agent_keys)
        model = _DummyCOMAModel(agent_keys)
        callback = MultiAgentBaseCallback()

        with tempfile.TemporaryDirectory() as tmpdir:
            coma_cfg = _make_learner_config(tmpdir, use_huber_loss=True, huber_delta=10.0)
            coma_learner = COMA_Learner(coma_cfg, grouping, model, callback)
            self.assertIsNone(coma_learner.huber_loss)

            huber_cfg = _make_learner_config(tmpdir, use_huber_loss=True, huber_delta=5.0)
            huber_learner = _DummyOnPolicyMARLLearner(huber_cfg, grouping, model, callback)
            self.assertIsInstance(huber_learner.huber_loss, torch.nn.HuberLoss)
            self.assertEqual(huber_learner.huber_loss.delta, 5.0)

    def test_scatter_add_1d_3d_empty_and_grad(self):
        src_1d = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        idx_1d = torch.tensor([0, 2, 0], dtype=torch.long)
        out_1d = scatter_add(src_1d, idx_1d, dim=0, dim_size=4)
        self.assertTrue(torch.allclose(out_1d, torch.tensor([4.0, 0.0, 2.0, 0.0])))
        out_1d.sum().backward()
        self.assertTrue(torch.allclose(src_1d.grad, torch.ones(3)))

        src_3d = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4).requires_grad_(True)
        idx_3d = torch.tensor([1, 0, 1], dtype=torch.long)
        out_3d = scatter_add(src_3d, idx_3d, dim=1, dim_size=3)
        expected_3d = torch.zeros(2, 3, 4, dtype=torch.float64)
        expected_3d[:, 0, :] = src_3d[:, 1, :]
        expected_3d[:, 1, :] = src_3d[:, 0, :] + src_3d[:, 2, :]
        self.assertTrue(torch.allclose(out_3d, expected_3d))
        out_3d.sum().backward()
        self.assertTrue(torch.allclose(src_3d.grad, torch.ones_like(src_3d)))

        empty_src = torch.zeros(2, 0, 4, dtype=torch.float32)
        empty_idx = torch.zeros(0, dtype=torch.long)
        empty_out = scatter_add(empty_src, empty_idx, dim=1, dim_size=3)
        self.assertEqual(tuple(empty_out.shape), (2, 3, 4))
        self.assertTrue(torch.equal(empty_out, torch.zeros(2, 3, 4)))

    def test_coordination_graph_degrees(self):
        expected = {
            "FULL": [2.0, 2.0, 2.0],
            "CYCLE": [2.0, 2.0, 2.0],
            "LINE": [1.0, 2.0, 1.0],
            "STAR": [2.0, 1.0, 1.0],
            "VDN": [0.0, 0.0, 0.0],
        }
        for graph_type, degrees in expected.items():
            graph = Coordination_Graph(n_vertexes=3, graph_type=graph_type, device="cpu")
            graph.set_coordination_graph()
            self.assertTrue(torch.allclose(graph.edges_n_in, torch.tensor(degrees)), msg=graph_type)

    def test_coma_benchmark_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_marl_smoke_args(tmpdir, buffer_size=20)
            runner = get_runner(algo="coma", env="mpe", env_id="simple_spread_v3", parser_args=args)
            runner.run(mode="benchmark")

    def test_dcg_benchmark_and_action_mask_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_marl_smoke_args(tmpdir)
            runner = get_runner(algo="dcg", env="mpe", env_id="simple_spread_v3", parser_args=args)
            runner.run(mode="benchmark")

            masked_args = _make_marl_smoke_args(tmpdir, use_actions_mask=True)
            masked_runner = get_runner(algo="dcg", env="mpe", env_id="simple_spread_v3", parser_args=masked_args)
            agent = masked_runner.agent
            obs_list = [
                {k: np.zeros(masked_runner.envs.observation_space[k].shape, dtype=np.float32) for k in agent.agent_keys}
                for _ in range(2)
            ]
            avail_list = [
                {k: np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32) for k in agent.agent_keys}
                for _ in range(2)
            ]
            out = agent.get_actions(obs_list=obs_list, avail_actions_list=avail_list, test_mode=True)
            for env_actions in out.env_actions:
                for act in env_actions.values():
                    self.assertEqual(int(act), 2)
            agent.finish()


if __name__ == "__main__":
    unittest.main()
