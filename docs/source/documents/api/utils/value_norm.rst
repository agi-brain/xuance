ValueNorm
========================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.utils.value_norm.ValueNorm(input_shape, norm_axes, beta, per_element_update, epsilon)

  :param input_shape: xxxxxx.
  :type input_shape: xxxxxx
  :param norm_axes: xxxxxx.
  :type norm_axes: xxxxxx
  :param beta: xxxxxx.
  :type beta: xxxxxx
  :param per_element_update: xxxxxx.
  :type per_element_update: xxxxxx
  :param epsilon: xxxxxx.
  :type epsilon: xxxxxx

.. py:function::
  xuance.torch.utils.value_norm.ValueNorm.reset_parameters()

  xxxxxx.

.. py:function::
  xuance.torch.utils.value_norm.ValueNorm.running_mean_var()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.value_norm.ValueNorm.update(input_vector)

  xxxxxx.

  :param input_vector: xxxxxx.
  :type input_vector: xxxxxx

.. py:function::
  xuance.torch.utils.value_norm.ValueNorm.normalize(input_vector)

  xxxxxx.

  :param input_vector: xxxxxx.
  :type input_vector: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.value_norm.ValueNorm.denormalize(input_vector)

  xxxxxx.

  :param input_vector: xxxxxx.
  :type input_vector: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import numpy as np
        import torch
        import torch.nn as nn


        class ValueNorm(nn.Module):
            """ Normalize a vector of observations - across the first norm_axes dimensions"""

            def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
                super(ValueNorm, self).__init__()

                self.input_shape = input_shape
                self.norm_axes = norm_axes
                self.epsilon = epsilon
                self.beta = beta
                self.per_element_update = per_element_update

                self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
                self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
                self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

                self.reset_parameters()

            def reset_parameters(self):
                self.running_mean.zero_()
                self.running_mean_sq.zero_()
                self.debiasing_term.zero_()

            def running_mean_var(self):
                debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
                debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
                debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
                return debiased_mean, debiased_var

            @torch.no_grad()
            def update(self, input_vector):
                if type(input_vector) == np.ndarray:
                    input_vector = torch.from_numpy(input_vector)
                input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

                batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
                batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

                if self.per_element_update:
                    batch_size = np.prod(input_vector.size()[:self.norm_axes])
                    weight = self.beta ** batch_size
                else:
                    weight = self.beta

                self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
                self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
                self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

            def normalize(self, input_vector):
                # Make sure input is float32
                if type(input_vector) == np.ndarray:
                    input_vector = torch.from_numpy(input_vector)
                input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

                mean, var = self.running_mean_var()
                out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

                return out

            def denormalize(self, input_vector):
                """ Transform normalized data back into original distribution """
                if type(input_vector) == np.ndarray:
                    input_vector = torch.from_numpy(input_vector)
                input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

                mean, var = self.running_mean_var()
                out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

                out = out.cpu().numpy()

                return out


  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python
