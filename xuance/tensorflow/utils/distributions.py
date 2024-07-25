from tensorflow.keras.activations import softplus
from abc import ABC, abstractmethod
from xuance.tensorflow import tf, tfd, Tensor


Categorical = tfd.Categorical
kl_div = tfd.kl_divergence


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
        self.probs, self.logits = None, None
        self.action_dim = action_dim

    def set_param(self, probs=None, logits=None):
        if probs is not None:
            self.distribution = tfd.Categorical(probs=probs)
        elif logits is not None:
            self.distribution = tfd.Categorical(logits=logits)
        else:
            raise RuntimeError("Failed to setup distributions without given probs or logits.")
        self.probs = self.distribution.probs
        self.logits = self.distribution.logits

    def get_param(self):
        return self.logits

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def entropy(self):
        return self.distribution.entropy()

    def stochastic_sample(self):
        return self.distribution.sample()

    def deterministic_sample(self):
        return tf.argmax(self.distribution.probs, dim=1)

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          CategoricalDistribution), "KL Divergence should be measured by two same distribution with the same type"
        return kl_div(self.distribution, other.distribution)


class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.mu, self.std = None, None
        self.action_dim = action_dim

    def set_param(self, mu, std):
        self.mu = mu
        self.std = std
        self.distribution = tfd.Normal(mu, std)

    def get_param(self):
        return self.mu, self.std

    def log_prob(self, x):
        return tf.math.reduce_sum(self.distribution.log_prob(x), axis=-1)

    def entropy(self):
        return tf.math.reduce_sum(self.distribution.entropy(), axis=-1)

    def stochastic_sample(self):
        return self.distribution.sample()

    def deterministic_sample(self):
        return self.mu

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          DiagGaussianDistribution), "KL Divergence should be measured by two same distribution with the same type"
        return kl_div(self.distribution, other.distribution)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_action):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_action

    def activated_rsample(self):
        return self.activation_fn(self.stochastic_sample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.stochastic_sample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.distribution.log_prob(act_pre_activated)
        correction = - 2. * (tf.math.log(2.0) - act_pre_activated - softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, tf.math.reduce_sum(log_prob, axis=-1)
