import torch
from torch import Tensor
from torch.nn.functional import softplus
from torch.distributions import Categorical
from torch.distributions import Normal
from abc import ABC, abstractmethod

kl_div = torch.distributions.kl_divergence


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
    def log_prob(self, x: torch.Tensor):
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
            self.distribution = Categorical(probs=probs, logits=logits)
        elif logits is not None:
            self.distribution = Categorical(probs=probs, logits=logits)
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
        return torch.argmax(self.distribution.probs, dim=1)

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
        self.distribution = Normal(mu, std)

    def get_param(self):
        return self.mu, self.std

    def log_prob(self, x):
        return self.distribution.log_prob(x).sum(-1)

    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def stochastic_sample(self):
        return self.distribution.sample()

    def rsample(self):
        return self.distribution.rsample()

    def deterministic_sample(self):
        return self.mu

    def kl_divergence(self, other: Distribution):
        assert isinstance(other,
                          DiagGaussianDistribution), "KL Divergence should be measured by two same distribution with the same type"
        return kl_div(self.distribution, other.distribution)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_action, device):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_action()
        self.device = device

    def activated_rsample(self):
        return self.activation_fn(self.rsample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.rsample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.distribution.log_prob(act_pre_activated)
        correction = - 2. * (torch.log(Tensor([2.0])).to(self.device) - act_pre_activated - softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, log_prob.sum(-1)
