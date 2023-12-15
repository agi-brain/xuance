Distributions
=================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.utils.distributions.Distribution()

.. py:function::
  xuance.torch.utils.distributions.Distribution.set_param(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.Distribution.get_param()

  xxxxxx.

.. py:function::
  xuance.torch.utils.distributions.Distribution.log_prob(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.Distribution.entropy()

  xxxxxx.

.. py:function::
  xuance.torch.utils.distributions.Distribution.stochastic_sample()

  xxxxxx.

.. py:function::
  xuance.torch.utils.distributions.Distribution.deterministic_sample()

  xxxxxx.

.. py:class::
  xuance.torch.utils.distributions.CategoricalDistribution()

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.__init__(action_dim)

  xxxxxx.

  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.set_param(logits)

  xxxxxx.

  :param logits: xxxxxx.
  :type logits: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.get_param()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.log_prob(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.entropy()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.stochastic_sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.deterministic_sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.kl_divergence(other)

  xxxxxx.

  :param other: xxxxxx.
  :type other: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.torch.utils.distributions.DiagGaussianDistribution()

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.__init__(action_dim)

  xxxxxx.

  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.set_param(mu, std)

  xxxxxx.

  :param mu: xxxxxx.
  :type mu: xxxxxx
  :param std: xxxxxx.
  :type std: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.get_param()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.log_prob(x)

  xxxxxx.

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.entropy()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.stochastic_sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.rsample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxxs

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.deterministic_sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxxs

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.kl_divergences(other)

  xxxxxx.

  :param other: xxxxxx.
  :type other: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.utils.distributions.Distribution()

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.set_param(args)

  :param args: xxxxxx.
  :type args: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.get_param()

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.log_prob(x)

  :param x: xxxxxx.
  :type x: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.entropy()

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.stochastic_sample()

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.deterministic_sample()

.. py:class::
  xuance.mindspore.utils.distributions.CategoricalDistribution(action_dim)

  :param action_dim: xxxxxx.
  :type action_dim: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.set_param(logits)

  :param logits: xxxxxx.
  :type logits: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.get_param()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.log_prob(x)

  :param x: xxxxxx.
  :type x: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.entropy()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.stochastic_sample()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.deterministic_sample()

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.kl_divergence(other)

  :param other: xxxxxx.
  :type other: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import torch
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

            def set_param(self, logits):
                self.logits = logits
                self.distribution = Categorical(logits=logits)

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

  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

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


