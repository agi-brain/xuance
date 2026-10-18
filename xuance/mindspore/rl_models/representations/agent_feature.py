from mindspore import Tensor
from mindspore.nn import Cell as Module
from xuance.mindspore.rl_models.modules.outputs import RepresentationOutput
from xuance.mindspore.rl_models.modules.identity_encoder import IdentityEncoder, IdentityFeatureFusion


class AgentFeatureEncoder(Module):
    def __init__(self,
                 representation: Module,
                 identity_encoder: IdentityEncoder,
                 fusion: IdentityFeatureFusion):
        super().__init__()

        self.obs_representation = representation
        self.identity_representation = identity_encoder
        self.fusion = fusion

        self.output_shapes = {
            "state": (fusion.output_dim,)
        }

    def construct(self,
                  observations: Tensor,
                  agent_indices=None,
                  **representation_kwargs) -> RepresentationOutput:
        obs_embeddings, rnn_states = self.obs_representation(observations, **representation_kwargs)

        identity_embeddings = self.identity_representation(agent_indices=agent_indices)

        fused_embeddings = self.fusion(
            observation_features=obs_embeddings,
            identity_features=identity_embeddings
        )

        # aux = dict(representation_output.aux)
        # aux = {'identity': identity_embeddings}

        return fused_embeddings, rnn_states
