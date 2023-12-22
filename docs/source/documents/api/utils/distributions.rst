Distributions
=================================

This module defines implementations of probability distributions for use in DRL algorithms.

.. raw:: html

    <br><hr>

PyTorch
---------------------------------------

.. py:class::
  xuance.torch.utils.distributions.Distribution()

  An abstract base class (ABC) that defines the interface for probability distributions used in RL algorithms.

.. py:function::
  xuance.torch.utils.distributions.Distribution.set_param(*args)

  A method that sets the parameters of a probability distribution.

  :param args: arguments for setting the distribution.
  :type args: tuple

.. py:function::
  xuance.torch.utils.distributions.Distribution.get_param()

  A method that gets the parameters of a probability distribution.

.. py:function::
  xuance.torch.utils.distributions.Distribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input probabilities.
  :type x: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.Distribution.entropy()

  A method that calculate the self entropy of the probability distribution.

.. py:function::
  xuance.torch.utils.distributions.Distribution.stochastic_sample()

  A method that randomly sample data from the distribution.

.. py:function::
  xuance.torch.utils.distributions.Distribution.deterministic_sample()

  A method that sample deterministic data from the distribution.


.. py:class::
  xuance.torch.utils.distributions.CategoricalDistribution(action_dim)

  Inherits from the Distribution base class and implements methods specific to categorical distributions.

  :param action_dim: The dimension of the action input.
  :type action_dim: int

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.set_param(logits)

  Set probability parameters.

  :param logits: The logits for categorical distributions.
  :type logits: Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.get_param()

  Set probability parameters.

  :return: probability parameters.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input probabilities.
  :type x: torch.Tensor
  :return: The log probability of input probabilities.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.entropy()

  A method that calculate the self entropy of the probability distribution.

  :return: the self entropy of the probability distribution.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.stochastic_sample()

  A method that randomly sample data from the distribution.

  :return: sampled data.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.deterministic_sample()

  A method that sample deterministic data from the distribution.

  :return: deterministic sampled data from the distribution.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.CategoricalDistribution.kl_divergence(other)

  A method that calculate the KL divergence between the two probability distribution.

  :param other: the other distribution.
  :return: the KL divergence between the two probability distribution.
  :rtype: torch.Tensor

.. py:class::
  xuance.torch.utils.distributions.DiagGaussianDistribution(action_dim)

  A diagonal Gaussian (normal) distribution. 
  This type of distribution is commonly used in DRL for continuous action spaces.

  :param action_dim: The dimension of the action input.
  :type action_dim: int

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.set_param(mu, std)

  Initializes the distribution using a PyTorch Normal distribution.

  :param mu: Mean value.
  :type mu: np.ndarray, torch.Tensor
  :param std: standard deviation.
  :type std: np.ndarray, torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.get_param()

  Returns the mean and standard deviation.

  :return: the mean and standard deviation.
  :rtype: tuple

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.log_prob(x)

  Computes the log probability of a given tensor x. It sums the result along the last dimension.

  :param x: The input tensor.
  :type x: torch.Tensor
  :return: the log probability of the given tensor x.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.entropy()

  Calculates the entropy of the distribution and sums the result along the last dimension.

  :return: the entropy of the distribution and sums the result along the last dimension.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.stochastic_sample()

  Generates a sample from the distribution using the sample method of the PyTorch Normal distribution.

  :return: a sample from the distribution.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.rsample()

  Uses the reparameterization trick to generate a sample from the distribution.

  :return: a sample from the distribution.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.deterministic_sample()

  This method returns the mean of the distribution.

  :return: the mean of the distribution.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.distributions.DiagGaussianDistribution.kl_divergences(other)

  This method computes the KL divergence between two distributions of the same type. 
  It asserts that the input distribution is of the correct type (DiagGaussianDistribution).

  :param other: the other distribution.
  :return: the KL divergence between two distributions.
  :rtype: torch.Tensor

.. raw:: html

    <br><hr>

TensorFlow
-------------------------------------------------------------

.. py:class::
  xuance.tensorflow.utils.distributions.Distribution()

  An abstract base class (ABC) that defines the interface for probability distributions used in RL algorithms.

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.set_param(*args)

  A method that sets the parameters of a probability distribution.

  :param args: arguments for setting the distribution.
  :type args: tuple

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.get_param()

  A method that gets the parameters of a probability distribution.

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input probabilities.
  :type x: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.entropy()

  A method that calculate the self entropy of the probability distribution.

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.stochastic_sample()

  A method that randomly sample data from the distribution.

.. py:function::
  xuance.tensorflow.utils.distributions.Distribution.deterministic_sample()

  A method that sample deterministic data from the distribution.

.. py:class::
  xuance.tensorflow.utils.distributions.CategoricalDistribution(action_dim)

  Inherits from the Distribution base class and implements methods specific to categorical distributions.

  :param action_dim: The dimension of the action input.
  :type action_dim: int

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.set_param(logits)

  Get probability parameters.

  :param logits: The logits for categorical distributions.
  :type logits: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.get_param()

  Set probability parameters.

  :return: probability parameters.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: The log probability of input probabilities.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.entropy()

  A method that calculate the self entropy of the probability distribution.

  :return: the self entropy of the probability distribution.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.stochastic_sample()

  A method that randomly sample data from the distribution.

  :return: sampled data.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.deterministic_sample()

  A method that sample deterministic data from the distribution.

  :return: deterministic sampled data from the distribution.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.CategoricalDistribution.kl_divergence(other)

  A method that calculate the KL divergence between the two probability distribution.

  :param other: the other distribution.
  :return: the KL divergence between the two probability distribution.
  :rtype: tf.Tensor

.. py:class::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution(action_dim)

  A diagonal Gaussian (normal) distribution. This type of distribution is commonly used in DRL for continuous action spaces.

  :param action_dim: The dimension of the action input.
  :type action_dim: int

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.set_param(mu, std)

  Initializes the distribution using a Normal distribution.

  :param mu: Mean value.
  :type mu: tf.Tensor
  :param std: standard deviation.
  :type std: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.get_param()

  Returns the mean and standard deviation.

  :return: the mean and standard deviation.
  :rtype: tuple

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.log_prob(x)

  Computes the log probability of a given tensor x. It sums the result along the last dimension.

  :param x: The input tensor.
  :type x: tf.Tensor
  :return: the log probability of the given tensor x.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.entropy()

  Calculates the entropy of the distribution and sums the result along the last dimension.

  :return: the entropy of the distribution and sums the result along the last dimension.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.stochastic_sample()

  Generates a sample from the distribution using the sample method of the PyTorch Normal distribution.

  :return: a sample from the distribution.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.rsample()

  Uses the reparameterization trick to generate a sample from the distribution.

  :return: a sample from the distribution.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.deterministic_sample()

  This method returns the mean of the distribution.

  :return: the mean of the distribution.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.distributions.DiagGaussianDistribution.kl_divergences(other)

  This method computes the KL divergence between two distributions of the same type. 
  It asserts that the input distribution is of the correct type (DiagGaussianDistribution).

  :param other: the other distribution.
  :return: the KL divergence between two distributions.
  :rtype: tf.Tensor

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------------------------

.. py:class::
  xuance.mindspore.utils.distributions.Distribution()

  An abstract base class (ABC) that defines the interface for probability distributions used in RL algorithms.

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.set_param(args)

  A method that sets the parameters of a probability distribution

  :param args: arguments for setting the distribution.
  :type args: tuple

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.get_param()

  A method that gets the parameters of a probability distribution.

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input tensor.
  :type x: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.entropy()

  A method that calculate the self entropy of the probability distribution.

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.stochastic_sample()

  A method that randomly sample data from the distribution.

.. py:function::
  xuance.mindspore.utils.distributions.Distribution.deterministic_sample()

  A method that sample deterministic data from the distribution.

.. py:class::
  xuance.mindspore.utils.distributions.CategoricalDistribution(action_dim)

  Inherits from the Distribution base class and implements methods specific to categorical distributions

  :param action_dim: The dimension of the action input.
  :type action_dim: int

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.set_param(logits)

  Set probability parameters.

  :param logits: The logits for categorical distributions.
  :type logits: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.get_param()

  Get probability parameters

  :return: probability parameters.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.log_prob(x)

  A method that calculate the log probability of input probabilities.

  :param x: The input tensor.
  :type x: ms.Tensor
  :return: The log probability of input probabilities.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.entropy()

  A method that calculate the self entropy of the probability distribution.

  :return: the self entropy of the probability distribution.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.stochastic_sample()

  A method that randomly sample data from the distribution.

  :return: sampled data.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.deterministic_sample()

  A method that sample deterministic data from the distribution.

  :return: deterministic sampled data from the distribution.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.distributions.CategoricalDistribution.kl_divergence(other)

  A method that calculate the KL divergence between the two probability distribution.

  :param other: the other distribution.
  :return: the KL divergence between the two probability distribution.
  :rtype: ms.Tensor

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

        import tensorflow_probability as tfp
        import tensorflow as tf

        tfd = tfp.distributions
        kl_div = tfd.kl_divergence
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
            def log_prob(self, x: tf.Tensor):
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
                self.distribution = tfd.Categorical(logits=logits)

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


