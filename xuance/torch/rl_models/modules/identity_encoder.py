import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Union, Literal


class IdentityEncoder(nn.Module):
    def __init__(self,
                 num_identities: int,
                 mode: Literal["none", "one_hot", "embedding"] = "none",
                 embedding_dim: Optional[int] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super().__init__()

        self.num_identities = num_identities
        self.mode = mode

        if mode == "none":
            self.output_dim = 0
            self.embedding = None
        elif mode == "one_hot":
            self.output_dim = num_identities
            self.embedding = None
        elif mode == "embedding":
            if embedding_dim is None:
                raise ValueError("embedding_dim is required when mode='embedding'.")

            self.output_dim = embedding_dim
            self.embedding = nn.Embedding(
                num_embeddings=num_identities,
                embedding_dim=embedding_dim,
                device=device
            )
        else:
            raise ValueError(f"Unsupported identity mode: {mode}.")

    def forward(self, agent_indices: Tensor) -> Optional[Tensor]:
        if self.mode == "none":
            encoded_identities = None
        elif self.mode == "one_hot":  # batch * 1 -> batch * num_classes
            encoded_identities = F.one_hot(agent_indices, num_classes=self.num_identities).squeeze(-2).float()
        elif self.mode == "embedding":
            encoded_identities = self.embedding(agent_indices).squeeze(-2)
        else:
            raise ValueError(f"Unsupported identity encoding: {self.mode}.")
        return encoded_identities


class IdentityFeatureFusion(nn.Module):
    def __init__(self,
                 observation_feature_dim: int,
                 identity_feature_dim: int,
                 mode: Literal["concat", "add", "film"] = "concat"):
        super().__init__()
        self.observation_feature_dim = observation_feature_dim
        self.identity_feature_dim = identity_feature_dim
        self.mode = mode

        if identity_feature_dim == 0:
            self.output_dim = observation_feature_dim
        elif mode == "concat":
            self.output_dim = observation_feature_dim + identity_feature_dim
        elif mode == "add":
            self.output_dim = observation_feature_dim
            self.identity_projection = nn.Linear(
                identity_feature_dim,
                observation_feature_dim
            )
        elif mode == "film":
            self.output_dim = observation_feature_dim
            self.modulation = nn.Linear(
                identity_feature_dim,
                observation_feature_dim * 2
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self,
                observation_features: Tensor,
                identity_features: Optional[Tensor]) -> Tensor:
        if identity_features is None:
            return observation_features
        elif self.mode == "concat":
            return torch.cat([observation_features, identity_features], dim=-1)
        elif self.mode == "add":
            return observation_features + self.identity_projection(identity_features)
        elif self.mode == "film":
            gamma, beta = self.modulation(identity_features).chunk(2, dim=-1)
            return (1.0 + gamma) * observation_features + beta
        else:
            raise RuntimeError("Invalid fusion mode.")


def build_identity_encoder(num_identities: int,
                           mode: Literal["none", "one_hot", "embedding"] = "none",
                           embedding_dim: Optional[int] = None,
                           device: Optional[Union[str, int, torch.device]] = None) -> IdentityEncoder:
    resolved_mode = "none" if num_identities == 1 else mode
    return IdentityEncoder(
        num_identities=num_identities,
        mode=resolved_mode,
        embedding_dim=embedding_dim,
        device=device
    )
