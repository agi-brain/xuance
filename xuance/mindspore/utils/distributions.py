import numpy as np
from abc import ABC, abstractmethod
from mindspore.nn.probability.distribution import Categorical, Normal
from xuance.mindspore import ms, ops, Tensor


def split_distributions(distribution):
    return_list = []
    if isinstance(distribution, CategoricalDistribution):
        shape = distribution.probs.shape
        probs = distribution.probs.view(-1, shape[-1])
        for prob in probs:
            dist = CategoricalDistribution(probs.shape[-1])
            dist.set_param(prob.unsqueeze(0))
            return_list.append(dist)
    elif isinstance(distribution, DiagGaussianDistribution):
        shape = distribution.mu.shape
        means = distribution.mu.view(-1, shape[-1])
        std = distribution.std
        for mu in means:
            dist = DiagGaussianDistribution(shape[-1])
            dist.set_param(mu, std)
            return_list.append(dist)
    else:
        raise NotImplementedError
    return np.array(return_list).reshape(shape[:-1])


def merge_distributions(distribution_list):
    if isinstance(distribution_list[0], CategoricalDistribution):
        probs = ms.ops.concat([dist.probs for dist in distribution_list], 0)
        action_dim = probs.shape[-1]
        dist = CategoricalDistribution(action_dim)
        dist.set_param(probs)
        return dist
    elif isinstance(distribution_list[0], DiagGaussianDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        mu = ops.cat([dist.mu for dist in distribution_list])
        std = ops.cat([dist.std for dist in distribution_list])
        action_dim = distribution_list[0].mu.shape[-1]
        dist = DiagGaussianDistribution(action_dim)
        mu = mu.view(shape + (action_dim,))
        std = std.view(shape + (action_dim,))
        dist.set_param(mu, std)
        return dist
    else:
        raise NotImplementedError


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
    def log_prob(self, x: ms.Tensor):
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

    def set_param(self, probs=None, logits=None):
        if probs is not None:
            self.distribution = Categorical(probs=probs)
            self.probs = probs
        elif logits is not None:
            probs = ops.softmax(logits, axis=-1)
            self.distribution = Categorical(probs=probs)
            self.logits = logits
        else:
            raise RuntimeError("Failed to setup distributions without given probs or logits.")

    def get_param(self):
        return self.logits

    def log_prob(self, x):
        return self.distribution.log_prob(value=Tensor(x))

    def entropy(self):
        return self.distribution.entropy()

    def stochastic_sample(self):
        return self.distribution.sample()

    def deterministic_sample(self):
        return self.argmax(self.distribution.probs)

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          CategoricalDistribution), "KL Divergence should be measured by two same distribution with the same type"
        return self.distribution.kl_loss(self.distribution, other.distribution)


class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.mu, self.std = None, None
        self.action_dim = action_dim

    def set_param(self, mu, std):
        self.mu = mu
        self.std = std
        self.distribution = Normal(mean=self.mu, sd=self.std, dtype=ms.float32)

    def get_param(self):
        return self.mu, self.std

    def log_prob(self, x: ms.Tensor):
        return self.distribution.log_prob(value=Tensor(x)).sum(-1)

    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def stochastic_sample(self):
        return self.distribution.sample()

    def deterministic_sample(self):
        return self.mu

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          DiagGaussianDistribution), "KL Divergence should be measured by two same distribution with the same type"
        return ops.kl_div(self.distribution, other.distribution)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_action):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_action()

    def activated_rsample(self):
        return self.activation_fn(self.stochastic_sample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.stochastic_sample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.distribution.log_prob(act_pre_activated)
        correction = - 2. * (ops.log(Tensor([2.0])) - act_pre_activated - ops.softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, log_prob.sum(-1)

