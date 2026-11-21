# Test module exports and output dataclass type annotations for PyTorch RL models.

import types
import typing
import unittest

import torch
from xuance.torch.rl_models.modules import (
    ActionOutput,
    CategoricalDistribution,
    Distribution,
    MARLActionOutput,
    ModelOutput,
    MultiAgentModelOutput,
    RepresentationOutput,
    StochasticActorOutput,
)


class TestRLModelsModules(unittest.TestCase):
    """Verify module exports and runtime-valid type hints on output dataclasses."""

    def test_distribution_export(self):
        self.assertTrue(issubclass(CategoricalDistribution, Distribution))

    def test_output_distribution_type_hints(self):
        for cls in (
            StochasticActorOutput,
            ModelOutput,
            MultiAgentModelOutput,
            ActionOutput,
        ):
            hints = typing.get_type_hints(cls)
            dist_hint = hints["distributions"]
            self.assertNotIsInstance(dist_hint, types.ModuleType)
            for arg in typing.get_args(dist_hint):
                self.assertNotIsInstance(arg, types.ModuleType)

    def test_output_dataclasses_instantiation(self):
        model_out = ModelOutput()
        self.assertIsNone(model_out.distributions)

        ma_out = MultiAgentModelOutput()
        self.assertIsNone(ma_out.distributions)

        action_out = ActionOutput(env_actions=[0])
        self.assertIsNone(action_out.distributions)

        marl_out = MARLActionOutput(env_actions=[])
        self.assertEqual(marl_out.env_actions, [])

        rep_out = RepresentationOutput(embeddings=torch.zeros(1, 1))
        actor_out = StochasticActorOutput(representations=rep_out)
        self.assertIsNone(actor_out.distributions)


if __name__ == "__main__":
    unittest.main()
