import mindspore as ms
from mindspore.nn.probability.distribution import Categorical, Normal
from abc import ABC, abstractmethod
from xuance.mindspore import ops, Tensor

import numpy as np
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore import _checkparam as Validator
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.probability.distribution._utils.utils import check_sum_equal_one, check_rank
from mindspore.nn.probability.distribution._utils.custom_ops import exp_generic, log_generic, broadcast_to, log_generic_with_check


def check_prob(p):
    """
    Check if p is a proper probability, i.e. 0 < p <1.

    Args:
        p (Tensor, Parameter): value to be checked.

    Raises:
        ValueError: if p is not a proper probability.
    """
    if p is None:
        raise ValueError(f'input value cannot be None in check_greater_zero')
    if isinstance(p, Parameter):
        if not isinstance(p.data, Tensor):
            return


class Categorical_MS(Categorical):
    def __init__(self,
                 probs=None,
                 seed=None,
                 dtype=mstype.int32,
                 name="Categorical"):
        param = dict(locals())
        param['param_dict'] = {'probs': probs}
        valid_dtype = mstype.uint_type + mstype.int_type + mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Categorical, self).__init__(seed, dtype, name, param)

        self._probs = self._add_parameter(probs, 'probs')
        if self.probs is not None:
            check_rank(self.probs)
            check_prob(self.probs)
            check_sum_equal_one(probs)

            # update is_scalar_batch and broadcast_shape
            # drop one dimension
            if self.probs.shape[:-1] == ():
                self._is_scalar_batch = True
            self._broadcast_shape = self._broadcast_shape[:-1]

        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.broadcast = broadcast_to
        self.cast = P.Cast()
        self.clip_by_value = ops.clip_by_value
        self.concat = P.Concat(-1)
        self.cumsum = P.CumSum()
        self.dtypeop = P.DType()
        self.exp = exp_generic
        self.expand_dim = P.ExpandDims()
        self.gather = P.GatherNd()
        self.greater = P.Greater()
        self.issubclass = inner.IsSubClass()
        self.less = P.Less()
        # when the graph kernel mode is enable
        # use Log directly as akg will handle the corner cases
        self.log = P.Log() if context.get_context("enable_graph_kernel") else log_generic
        self.log_with_check = P.Log() if context.get_context("enable_graph_kernel") else log_generic_with_check
        self.log_softmax = P.LogSoftmax()
        self.logicor = P.LogicalOr()
        self.logicand = P.LogicalAnd()
        self.multinomial = P.Multinomial(seed=self.seed)
        self.reshape = P.Reshape()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select = P.Select()
        self.shape = P.Shape()
        self.softmax = P.Softmax()
        self.squeeze = P.Squeeze()
        self.squeeze_first_axis = P.Squeeze(0)
        self.squeeze_last_axis = P.Squeeze(-1)
        self.square = P.Square()
        self.transpose = P.Transpose()

        self.index_type = mstype.int32
        self.nan = np.nan


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
        self.argmax = ops.Argmax(output_type=ms.int32, axis=1)

    def set_param(self, probs=None):
        self.distribution = Categorical_MS(probs=probs)
        self.probs = self.distribution.probs

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

