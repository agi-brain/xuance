import numpy as np
from tensorflow.keras.activations import softplus
from abc import abstractmethod
from xuance.tensorflow import tf, Tensor, tfd

Categorical = tfd.Categorical

class Distribution:
    def __init__(self):
        pass

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
        self.probs, self.logits = None, None
        self.action_dim = action_dim

    def set_param(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs / probs.sum(-1, keepdims=True)
            self.logits = tf.math.log(probs) - tf.math.log1p(-probs)
        elif logits is not None:
            self.logits = logits
            self.probs = tf.nn.softmax(logits, axis=-1)
        else:
            raise RuntimeError("Either probs or logits must be specified.")

    def get_param(self):
        return self.probs or self.logits

    def log_prob(self, x):
        x = tf.expand_dims(tf.cast(x, dtype=tf.int32), -1)
        log_probs = tf.nn.log_softmax(self.logits)
        y = tf.gather(log_probs, x, batch_dims=1)
        y = tf.squeeze(y, axis=-1)
        return y

    def entropy(self):
        log_probs = tf.nn.log_softmax(self.logits)
        e = -tf.reduce_sum(self.probs * log_probs, axis=-1, keepdims=True)
        return e

    def stochastic_sample(self):
        logits_detach = self.logits.numpy()
        sampled_actions = tf.random.categorical(logits_detach, num_samples=1)
        return tf.squeeze(sampled_actions, axis=-1)

    def deterministic_sample(self):
        return tf.argmax(self.probs, dim=1)

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          CategoricalDistribution), "KL Divergence should be measured by two same distribution with the same type"
        log_p = tf.nn.log_softmax(self.logits, axis=-1)  # log P(a)
        log_q = tf.nn.log_softmax(other.logits, axis=-1)  # log Q(a)
        p = tf.math.exp(log_p)  # P(a)
        kl = tf.reduce_sum(p * (log_p - log_q), axis=-1)
        return kl


class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.mu, self.std = None, None
        self.action_dim = action_dim

    def set_param(self, mu, std):
        self.mu = mu
        self.std = std

    def get_param(self):
        return self.mu, self.std

    def log_prob(self, x):
        log_std = tf.math.log(self.std + 1e-8)
        log_prob = -0.5 * (((x - self.mu) / (self.std + 1e-8)) ** 2 + 2.0 * log_std + tf.math.log(2.0 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=False)
        return log_prob

    def entropy(self):
        log_std = tf.math.log(self.std + 1e-8)
        entropy = tf.reduce_sum(0.5 + 0.5 * tf.math.log(2.0 * np.pi) + log_std, axis=-1, keepdims=True)
        return entropy

    def stochastic_sample(self):
        eps = tf.random.normal(shape=tf.shape(self.mu))  # 𝜖 ~ N(0, 1)
        action = self.mu + self.std * eps  # Reparameterization trick
        return action

    def deterministic_sample(self):
        return self.mu

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          DiagGaussianDistribution), "KL Divergence should be measured by two same distribution with the same type"
        var1 = tf.square(self.std)
        var2 = tf.square(other.std)
        kl = tf.math.log(other.std / self.std) + (var1 + tf.square(self.mu - other.mu)) / (2.0 * var2) - 0.5
        return tf.reduce_sum(kl, axis=-1)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_action):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_action

    def activated_rsample(self):
        return self.activation_fn(self.stochastic_sample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.stochastic_sample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.log_prob(act_pre_activated)
        correction = - 2. * (tf.math.log(2.0) - act_pre_activated - softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, tf.math.reduce_sum(log_prob, axis=-1)
