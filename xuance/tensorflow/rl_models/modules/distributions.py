import numpy as np
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import Tensor
from keras.activations import softplus


def split_distributions(distribution):
    """Splits a batch of distributions into individual instances.

    This function separates a batch of distributions (either `CategoricalDistribution`
    or `DiagGaussianDistribution`) into individual distribution objects.

    Args:
        distribution (CategoricalDistribution or DiagGaussianDistribution): The input
            distribution batch to be split.

    Returns:
        np.ndarray: A reshaped array of individual distribution instances.

    Raises:
        NotImplementedError: If the distribution type is not supported.
    """

    return_list = []
    if isinstance(distribution, CategoricalDistribution):
        shape = distribution.logits.shape
        logits = tf.reshape(distribution.logits, [-1, shape[-1]])
        for logit in logits:
            dist = CategoricalDistribution(logits.shape[-1])
            dist.set_param(logits=tf.stop_gradient(tf.expand_dims(logit, 0)))
            return_list.append(dist)
    elif isinstance(distribution, DiagGaussianDistribution):
        shape = distribution.mu.shape
        means = tf.reshape(distribution.mu, [-1, shape[-1]])
        std = distribution.std
        for mu in means:
            dist = DiagGaussianDistribution(shape[-1])
            dist.set_param(mu, std)
            return_list.append(dist)
    else:
        raise NotImplementedError
    return np.array(return_list).reshape(shape[:-1])


def merge_distributions(distribution_list):
    """Merges a list of individual distributions back into a batch distribution.

    This function reconstructs a batched distribution from a list (or array) of
    individual distributions, supporting both categorical and diagonal Gaussian distributions.

    Args:
        distribution_list (list or np.ndarray): A list or array of individual distribution instances.

    Returns:
        CategoricalDistribution or DiagGaussianDistribution: A merged batch distribution.

    Raises:
        NotImplementedError: If the distribution type is not supported.
    """

    if isinstance(distribution_list[0], CategoricalDistribution):
        logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        dist.set_param(logits=tf.stop_gradient(logits))
        return dist
    elif isinstance(distribution_list[0], DiagGaussianDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        mu = tf.concat([dist.mu for dist in distribution_list], axis=0)
        std = tf.concat([dist.std for dist in distribution_list], axis=0)
        action_dim = distribution_list[0].mu.shape[-1]
        dist = DiagGaussianDistribution(action_dim)
        mu = tf.reshape(mu, shape + (action_dim,))
        std = tf.reshape(std, shape + (action_dim,))
        dist.set_param(mu=mu, std=std)
        return dist
    elif isinstance(distribution_list[0, 0], CategoricalDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        logits = tf.reshape(logits, shape + (action_dim,))
        dist.set_param(tf.stop_gradient(logits))
        return dist
    else:
        pass


class Distribution(ABC):
    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def set_param(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_param(self):
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def entropy(self):
        raise NotImplementedError

    @abstractmethod
    def stochastic_sample(self):
        raise NotImplementedError

    @abstractmethod
    def deterministic_sample(self):
        raise NotImplementedError


class CategoricalDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim
        self.probs, self.logits = None, None

    def set_param(
            self,
            probs: tf.Tensor | None = None,
            logits: tf.Tensor | None = None
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of `probs` or `logits` must be specified.")

        if logits is not None:
            self.logits = tf.convert_to_tensor(logits)
            self.probs = tf.nn.softmax(self.logits, axis=-1)
        else:
            probs = tf.convert_to_tensor(probs)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)

            epsilon = tf.cast(tf.keras.backend.epsilon(), probs.dtype)
            self.probs = probs
            self.logits = tf.math.log(tf.clip_by_value(probs, epsilon, 1.0))

    def get_param(self):
        return self.logits

    def log_prob(self, x: Tensor) -> Tensor:
        x = tf.cast(x, tf.int32)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

    def entropy(self):
        log_probs = tf.nn.log_softmax(self.logits, axis=-1)
        probs = tf.exp(log_probs)
        return -tf.reduce_sum(probs * log_probs, axis=-1)

    def stochastic_sample(self):
        original_batch_shape = tf.shape(self.logits)[:-1]
        action_dim = tf.shape(self.logits)[-1]
        flat_logits = tf.reshape(
            self.logits,
            shape=(-1, action_dim),
        )

        samples = tf.random.categorical(logits=flat_logits, num_samples=1, dtype=tf.int32)
        return tf.reshape(samples, original_batch_shape)

    def deterministic_sample(self):
        return tf.argmax(self.probs, axis=1)

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          CategoricalDistribution), "KL Divergence should be measured by two same distribution with the same type"
        log_p = tf.nn.log_softmax(self.logits, axis=-1)  # log P(a)
        log_q = tf.nn.log_softmax(other.logits, axis=-1)  # log Q(a)
        p = tf.math.exp(log_p)  # P(a)
        kl = tf.reduce_sum(p * (log_p - log_q), axis=-1)
        return kl


class DiagGaussianDistribution(Distribution):
    def __init__(
            self,
            action_dim: int,
            min_std: float = 1e-6,
    ):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.min_std = min_std

        self.mu = None
        self.std = None
        self.log_std = None

    def set_param(self, mu: Tensor, std: Tensor):
        self.mu = mu
        self.std = std

        self.mu = tf.convert_to_tensor(mu)
        std = tf.cast(std, self.mu.dtype)

        min_std = tf.cast(self.min_std, self.mu.dtype)
        self.std = tf.maximum(std, min_std)
        self.log_std = tf.math.log(self.std)

    def get_param(self):
        return self.mu, self.std

    def log_prob_per_dimension(self, x: Tensor) -> Tensor:
        x = tf.cast(x, self.mu.dtype)
        log_two_pi = tf.math.log(tf.cast(2.0 * np.pi, self.mu.dtype))
        normalized = (x - self.mu) / self.std
        return -0.5 * (tf.square(normalized) + 2.0 * self.log_std + log_two_pi)

    def log_prob(self, x: Tensor) -> Tensor:
        return tf.reduce_sum(self.log_prob_per_dimension(x), axis=-1)

    def entropy(self) -> Tensor:
        log_two_pi = tf.math.log(tf.cast(2.0 * np.pi, self.mu.dtype))
        entropy_per_dimension = self.log_std + 0.5 * (1.0 + log_two_pi)
        return tf.reduce_sum(entropy_per_dimension, axis=-1)

    def rsample(self) -> Tensor:
        epsilon = tf.random.normal(shape=tf.shape(self.mu), dtype=self.mu.dtype)
        return self.mu + self.std * epsilon

    def stochastic_sample(self) -> Tensor:
        return tf.stop_gradient(self.rsample())

    def deterministic_sample(self) -> Tensor:
        return self.mu

    def kl_divergence(self, other: Distribution) -> Tensor:
        if not isinstance(other, DiagGaussianDistribution):
            raise TypeError("KL divergence requires another DiagGaussianDistribution.")

        other_std = tf.maximum(tf.cast(other.std, self.std.dtype), tf.cast(self.min_std, self.std.dtype))

        variance = tf.square(self.std)
        other_variance = tf.square(other_std)
        mean_difference = self.mu - tf.cast(other.mu, self.mu.dtype)
        kl_per_dimension = tf.math.log(other_std / self.std) + (variance + tf.square(mean_difference)) / (
                    2.0 * other_variance) - 0.5
        return tf.reduce_sum(kl_per_dimension, axis=-1)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_action):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_action

    def activated_rsample(self):
        return self.activation_fn(self.stochastic_sample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.rsample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.log_prob(act_pre_activated)
        log_prob = tf.expand_dims(log_prob, axis=-1)
        correction = - 2. * (tf.math.log(2.0) - act_pre_activated - softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, tf.math.reduce_sum(log_prob, axis=-1)
