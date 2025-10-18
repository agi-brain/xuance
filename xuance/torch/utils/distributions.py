import math
from numbers import Number
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.nn.functional import softplus, one_hot, softmax
from torch.distributions import Categorical, Bernoulli, Normal, constraints
from torch.distributions.utils import broadcast_all
from xuance.common import Callable
from xuance.torch import Tensor
from xuance.torch.utils.operations import sym_log, sym_exp

kl_div = torch.distributions.kl_divergence


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
        logits = distribution.logits.view(-1, shape[-1])
        for logit in logits:
            dist = CategoricalDistribution(logits.shape[-1])
            dist.set_param(logits=logit.unsqueeze(0).detach())
            return_list.append(dist)
    elif isinstance(distribution, DiagGaussianDistribution):
        shape = distribution.mu.shape
        means = distribution.mu.view(-1, shape[-1])
        std = distribution.std
        for mu in means:
            dist = DiagGaussianDistribution(shape[-1])
            dist.set_param(mu.detach(), std.detach())
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
        logits = torch.cat([dist.logits for dist in distribution_list], dim=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        dist.set_param(logits=logits.detach())
        return dist
    elif isinstance(distribution_list[0], DiagGaussianDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        mu = torch.cat([dist.mu for dist in distribution_list], dim=0)
        std = torch.cat([dist.std for dist in distribution_list], dim=0)
        action_dim = distribution_list[0].mu.shape[-1]
        dist = DiagGaussianDistribution(action_dim)
        mu = mu.view(shape + (action_dim,))
        std = std.view(shape + (action_dim,))
        dist.set_param(mu, std)
        return dist
    elif isinstance(distribution_list[0, 0], CategoricalDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        logits = torch.cat([dist.logits for dist in distribution_list], dim=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        logits = logits.view(shape + (action_dim,))
        dist.set_param(logits=logits.detach())
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


class SymLogDistribution:
    def __init__(self,
                 mode: Tensor,
                 dims: int,
                 dist: str = "mse",
                 agg: str = "sum",
                 tol: float = 1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims:]

    @property
    def mode(self) -> Tensor:
        return sym_exp(self._mode)

    @property
    def mean(self) -> Tensor:
        return sym_exp(self._mode)

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log probability of a value under this distribution.

        Args:
            value: The observed value (in original space) to evaluate.

        Returns:
            Log probability tensor, aggregated over event dimensions as specified.

        Raises:
            AssertionError: If value shape does not match the distribution's shape.
            NotImplementedError: If invalid distance or aggregation methods are provided.
        """
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - sym_log(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - sym_log(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MSEDistribution:
    def __init__(self, mode: Tensor, dims: int, agg: str = "sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        return self._mode

    @property
    def mean(self) -> Tensor:
        return self._mode

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = sym_log,
        transbwd: Callable[[Tensor], Tensor] = sym_exp,
    ):
        self.logits = logits
        self.probs = softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])  # logits.shape[-1] = 255 (len(self.bins))
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1  # shape: [1, ]
        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (one_hot(below, len(self.bins)) * weight_below[..., None]
                  + one_hot(above, len(self.bins)) * weight_above[..., None]).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=self.dims)


class BernoulliSafeMode(Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(torch.distributions.Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def entropy(self):
        return self._entropy


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale