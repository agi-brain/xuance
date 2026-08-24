from typing import Optional, Literal
from xuance.tensorflow import tf, keras, Tensor, Module


class IdentityEncoder(Module):
    def __init__(
            self,
            num_identities: int,
            mode: Literal["none", "one_hot", "embedding"] = "none",
            embedding_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_identities = num_identities
        self.mode = mode
        self.embedding_dim = embedding_dim

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
            self.embedding = keras.layers.Embedding(
                input_dim=num_identities,
                output_dim=embedding_dim
            )

        else:
            raise ValueError(f"Unsupported identity mode: {mode}.")

    def call(
            self,
            agent_indices: Tensor,
            training: Optional[bool] = None
    ) -> Optional[Tensor]:

        if self.mode == "none":
            encoded_identities = None

        elif self.mode == "one_hot":
            # batch * 1 -> batch * num_identities
            encoded_identities = tf.one_hot(agent_indices, depth=self.num_identities, dtype=tf.float32)
            encoded_identities = tf.squeeze(encoded_identities, axis=-2)

        elif self.mode == "embedding":
            encoded_identities = self.embedding(agent_indices)
            encoded_identities = tf.squeeze(encoded_identities, axis=-2)

        else:
            raise ValueError(f"Unsupported identity encoding: {self.mode}.")

        return encoded_identities

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            num_identities=self.num_identities,
            mode=self.mode,
            embedding_dim=self.embedding_dim
        ))
        return config


class IdentityFeatureFusion(Module):
    def __init__(
            self,
            observation_feature_dim: int,
            identity_feature_dim: int,
            mode: Literal["concat", "add", "film"] = "concat",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.observation_feature_dim = observation_feature_dim
        self.identity_feature_dim = identity_feature_dim
        self.mode = mode
        self.kwargs = kwargs

        if identity_feature_dim == 0:
            self.output_dim = observation_feature_dim

        elif mode == "concat":
            self.output_dim = observation_feature_dim + identity_feature_dim

        elif mode == "add":
            self.output_dim = observation_feature_dim
            self.identity_projection = keras.layers.Dense(
                units=observation_feature_dim
            )

        elif mode == "film":
            self.output_dim = observation_feature_dim

            self.modulation = keras.layers.Dense(
                units=observation_feature_dim * 2
            )

        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def call(
            self,
            observation_features: Tensor,
            identity_features: Optional[Tensor],
            training: Optional[bool] = None
    ) -> Tensor:

        if identity_features is None:
            return observation_features

        elif self.mode == "concat":
            return tf.concat([observation_features, identity_features], axis=-1)

        elif self.mode == "add":
            return observation_features + self.identity_projection(identity_features)

        elif self.mode == "film":
            modulation = self.modulation(identity_features)
            gamma, beta = tf.split(modulation, num_or_size_splits=2, axis=-1)
            return (1.0 + gamma) * observation_features + beta

        else:
            raise RuntimeError("Invalid fusion mode.")

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            observation_feature_dim=self.observation_feature_dim,
            identity_feature_dim=self.identity_feature_dim,
            mode=self.mode,
            kwargs=self.kwargs
        ))
        return config


def build_identity_encoder(
        num_identities: int,
        mode: Literal["none", "one_hot", "embedding"] = "none",
        embedding_dim: Optional[int] = None
) -> IdentityEncoder:
    resolved_mode = "none" if num_identities == 1 else mode

    return IdentityEncoder(
        num_identities=num_identities,
        mode=resolved_mode,
        embedding_dim=embedding_dim
    )
