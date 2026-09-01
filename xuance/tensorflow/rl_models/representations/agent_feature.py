from tensorflow import Tensor
from xuance.tensorflow.utils import Module
from xuance.tensorflow.rl_models.modules.outputs import RepresentationOutput
from xuance.tensorflow.rl_models.modules.identity_encoder import IdentityEncoder, IdentityFeatureFusion


class AgentFeatureEncoder(Module):
    def __init__(
            self,
            representation: Module,
            identity_encoder: IdentityEncoder,
            fusion: IdentityFeatureFusion,
            **kwargs
    ):
        super().__init__()

        self.obs_representation = representation
        self.identity_representation = identity_encoder
        self.fusion = fusion

        self.output_shapes = {
            "state": (fusion.output_dim,)
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "representation": self.obs_representation.clone(copy_weights=True, trainable=False,
                                                            name="target_representation"),
            "identity_encoder": self.identity_representation.clone(copy_weights=True, trainable=False,
                                                                   name="target_identity_encoder"),
            "fusion": self.fusion.clone(copy_weights=True, trainable=False, name="target_fusion")
        })
        return config

    def call(self,
             observations: Tensor,
             agent_indices=None,
             **representation_kwargs) -> RepresentationOutput:
        representation_output = self.obs_representation(observations, **representation_kwargs)

        identity_embeddings = self.identity_representation(agent_indices=agent_indices)

        fused_embeddings = self.fusion(
            observation_features=representation_output.embeddings,
            identity_features=identity_embeddings
        )

        aux = dict(representation_output.aux)
        aux['identity'] = identity_embeddings

        return RepresentationOutput(
            embeddings=fused_embeddings,
            rnn_states=representation_output.rnn_states,
            aux=aux
        )
