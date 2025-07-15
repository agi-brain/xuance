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
        return y

    def entropy(self):
        log_probs = tf.nn.log_softmax(self.logits)
        e = -tf.reduce_sum(self.probs * log_probs, axis=-1, keepdims=True)
        return e

    def stochastic_sample(self):
        logits_detach = self.logits.numpy()
        sampled_actions = tf.random.categorical(self.logits, num_samples=1)
        return tf.squeeze(sampled_actions, axis=-1)

    def deterministic_sample(self):
        return tf.argmax(self.probs, dim=1)

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
