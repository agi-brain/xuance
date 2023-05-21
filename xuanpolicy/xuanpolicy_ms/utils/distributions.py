import mindspore as ms
from mindspore.nn.probability.distribution import Categorical
from abc import ABC, abstractmethod


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

    def set_param(self, logits):
        self.logits = logits
        self.distribution = Categorical(probs=logits)

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
        return self.distribution.kl_loss(self.distribution, other.distribution)
