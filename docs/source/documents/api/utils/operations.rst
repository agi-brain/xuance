Operations
===========================================

This module provides utility functions related to DRL.

.. raw:: html

    <br><hr>

PyTorch
----------------------------------

.. py:function::
  xuance.torch.utils.operations.update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor)

  This function updates the learning rate of an optimizer with linear decay.

  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param step: Current step.
  :type step: int
  :param total_steps: Total number of steps.
  :type total_steps: int
  :param initial_lr: Initial learning rate.
  :type initial_lr: float
  :param end_factor: Factor for the minimum learning rate.
  :type end_factor: float

.. py:function::
  xuance.torch.utils.operations.set_seed(seed)

  This function sets random seeds for reproducibility.

  :param seed: Random seed.
  :type seed: int

.. py:function::
  xuance.torch.utils.operations.get_flat_grad(y, model)

  This function returns the flattened gradients of a tensor y with respect to the parameters of a PyTorch model.

  :param y: Input tensor.
  :type y: torch.Tensor
  :param model: PyTorch model
  :type model: nn.Module
  :return: the flattened gradients.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.operations.get_flat_params(model)

  This function returns the flattened parameters of a PyTorch model.

  :param model: PyTorch model.
  :type model: nn.Module
  :return: the flattened parameters of a PyTorch model.
  :rtype: torch.Tensor

.. py:function::
  xuance.torch.utils.operations.assign_from_flat_grads(flat_grads, model)

  This function assigns flattened gradients to the parameters of a PyTorch model.

  :param flat_grads: Flattened gradients.
  :type flat_grads: torch.Tensor
  :param model: PyTorch model.
  :type model: torch.Module

.. py:function::
  xuance.torch.utils.operations.assign_from_flat_params(flat_grads, model)

  This function assigns flattened parameters to the parameters of a PyTorch model.

  :param flat_grads: Flattened parameters.
  :type flat_grads: torch.Tensor
  :param model: PyTorch model.
  :type model: torch.Module

.. py:function::
  xuance.torch.utils.operations.split_distributions(distribution)

  This function splits a distribution into a list of distributions.

  :param distribution: Input distribution.
  :return: The splited distributions.

.. py:function::
  xuance.torch.utils.operations.merge_distributions(distribution_list)

  This function merges a list of distributions into a single distribution.

  :param distribution_list: Input distribution list.
  :type distribution_list: list
  :return: A merged distribution.

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:function::
  xuance.tensorflow.utils.operations.update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor)

  This function updates the learning rate of an optimizer with linear decay.

  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param step: Current step.
  :type step: int
  :param total_steps: Total number of steps.
  :type total_steps: int
  :param initial_lr: Initial learning rate.
  :type initial_lr: float
  :param end_factor: Factor for the minimum learning rate.
  :type end_factor: float

.. py:function::
  xuance.tensorflow.utils.operations.set_seed(seed)

  This function sets random seeds for reproducibility.

  :param seed: Random seed.
  :type seed: int

.. py:function::
  xuance.tensorflow.utils.operations.get_flat_params(model)

  This function returns the flattened gradients of a tensor y with respect to the parameters of a PyTorch model.

  :param model: Tensorflow keras model.
  :type model: tk.Model
  :return: the flattened parameters of a PyTorch model.
  :rtype: tf.Tensor

.. py:function::
  xuance.tensorflow.utils.operations.assign_from_flat_grads(flat_grads, model)

  This function assigns flattened gradients to the parameters of a model.

  :param flat_grads: Flattened gradients.
  :type flat_grads: tf.Tensor
  :param model: Tensorflow keras model.
  :type model: tk.Model

.. py:function::
  xuance.tensorflow.utils.operations.assign_from_flat_params(flat_grads, model)

  This function assigns flattened parameters to the parameters of a model.

  :param flat_grads: Flattened parameters.
  :type flat_grads: tf.Tensor
  :param model: Tensorflow keras model.
  :type model: tk.Model

.. py:function::
  xuance.tensorflow.utils.operations.split_distributions(distribution)

  This function splits a distribution into a list of distributions.

  :param distribution: Input distribution.
  :return: The splited distributions.

.. py:function::
  xuance.tensorflow.utils.operations.merge_distributions(distribution_list)

  This function merges a list of distributions into a single distribution.

  :param distribution_list: Input distribution list.
  :type distribution_list: list
  :return: A merged distribution.

.. raw:: html

    <br><hr>

MindSpore
----------------------------------------------

.. py:function::
  xuance.mindspore.utils.operations.update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor)

  This function updates the learning rate of an optimizer with linear decay.

  :param optimizer: The optimizer that update the parameters of the model.
  :type optimizer: Optimizer
  :param step: Current step.
  :type step: int
  :param total_steps: Total number of steps.
  :type total_steps: int
  :param initial_lr: Initial learning rate.
  :type initial_lr: float
  :param end_factor: Factor for the minimum learning rate.
  :type end_factor: float

.. py:function::
  xuance.mindspore.utils.operations.set_seed(seed)

  This function sets random seeds for reproducibility.

  :param seed: Random seed.
  :type seed: int

.. py:function::
  xuance.mindspore.utils.operations.get_flat_grad(y, model)

  This function returns the flattened gradients of a tensor y with respect to the parameters of a PyTorch model.

  :param y: Input tensor.
  :type y: tf.Tensor
  :param model: Mindspore model.
  :type model: ms.Cell
  :return: the flattened gradients.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.operations.get_flat_params(model)

  This function returns the flattened parameters of a mindspore model.

  :param model: Mindspore model.
  :type model: ms.Cell
  :return: the flattened parameters of a mindspore model.
  :rtype: ms.Tensor

.. py:function::
  xuance.mindspore.utils.operations.assign_from_flat_grads(flat_grads, model)

  This function assigns flattened parameters to the parameters of a mindspore model.

  :param flat_grads: Flattened parameters.
  :type flat_grads: ms.Tensor
  :param model: Mindspore model.
  :type model: ms.Cell

.. py:function::
  xuance.mindspore.utils.operations.assign_from_flat_params(flat_grads, model)

  This function assigns flattened parameters to the parameters of a mindspore model.

  :param flat_grads: Flattened parameters.
  :type flat_grads: ms.Tensor
  :param model: Mindspore model.
  :type model: ms.Cell

.. py:function::
  xuance.mindspore.utils.operations.split_distributions(distribution)

  This function splits a distribution into a list of distributions.

  :param distribution: Input distribution.
  :return: The splited distributions.

.. py:function::
  xuance.mindspore.utils.operations.merge_distributions(distribution_list)

  This function merges a list of distributions into a single distribution.

  :param distribution_list: Input distribution list.
  :type distribution_list: list
  :return: A merged distribution.

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import random

        import torch
        import torch.nn as nn
        import numpy as np
        from .distributions import CategoricalDistribution, DiagGaussianDistribution


        def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
            lr = initial_lr * (1 - step / float(total_steps))
            if lr < end_factor * initial_lr:
                lr = end_factor * initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        def set_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)


        def get_flat_grad(y: torch.Tensor, model: nn.Module) -> torch.Tensor:
            grads = torch.autograd.grad(y, model.parameters())
            return torch.cat([grad.reshape(-1) for grad in grads])


        def get_flat_params(model: nn.Module) -> torch.Tensor:
            params = model.parameters()
            return torch.cat([param.reshape(-1) for param in params])


        def assign_from_flat_grads(flat_grads: torch.Tensor, model: nn.Module) -> nn.Module:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def assign_from_flat_params(flat_params: torch.Tensor, model: nn.Module) -> nn.Module:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def split_distributions(distribution):
            return_list = []
            if isinstance(distribution, CategoricalDistribution):
                shape = distribution.logits.shape
                logits = distribution.logits.view(-1, shape[-1])
                for logit in logits:
                    dist = CategoricalDistribution(logits.shape[-1])
                    dist.set_param(logit.unsqueeze(0).detach())
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
            if isinstance(distribution_list[0], CategoricalDistribution):
                logits = torch.cat([dist.logits for dist in distribution_list], dim=0)
                action_dim = logits.shape[-1]
                dist = CategoricalDistribution(action_dim)
                dist.set_param(logits.detach())
                return dist
            elif isinstance(distribution_list[0], DiagGaussianDistribution):
                shape = distribution_list.shape
                distribution_list = distribution_list.reshape([-1])
                mu = torch.cat([dist.mu for dist in distribution_list], dim=0)
                std = torch.cat([dist.std for dist in distribution_list], dim=0)
                action_dim = distribution_list[0].mu.shape[-1]
                dist = DiagGaussianDistribution(action_dim)
                mu = mu.view(shape + (action_dim, ))
                std = std.view(shape + (action_dim,))
                dist.set_param(mu, std)
                return dist
            elif isinstance(distribution_list[0, 0], CategoricalDistribution):
                shape = distribution_list.shape
                distribution_list = distribution_list.reshape([-1])
                logits = torch.cat([dist.logits for dist in distribution_list], dim=0)
                action_dim = logits.shape[-1]
                dist = CategoricalDistribution(action_dim)
                logits = logits.view(shape + (action_dim, ))
                dist.set_param(logits.detach())
                return dist
            else:
                pass

  .. group-tab:: TensorFlow

    .. code-block:: python

        import random
        import numpy as np
        from .distributions import CategoricalDistribution, DiagGaussianDistribution
        import tensorflow as tf
        import tensorflow.keras as tk


        def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
            lr = initial_lr * (1 - step / float(total_steps))
            if lr < end_factor * initial_lr:
                lr = end_factor * initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        def set_seed(seed):
            tf.random.set_seed(seed)
            tf.compat.v1.set_random_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # def get_flat_grad(y: tf.Tensor, model: tk.Model) -> tf.Tensor:
        #     grads = torch.autograd.grad(y, model.parameters())
        #     return torch.cat([grad.reshape(-1) for grad in grads])


        def get_flat_params(model: tk.Model) -> tf.Tensor:
            params = model.parameters()
            return tf.concat([param.reshape(-1) for param in params])


        def assign_from_flat_grads(flat_grads: tf.Tensor, model: tk.Model) -> tk.Model:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def assign_from_flat_params(flat_params: tf.Tensor, model: tk.Model) -> tk.Model:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def split_distributions(distribution):
            return_list = []
            if isinstance(distribution, CategoricalDistribution):
                shape = distribution.logits.shape
                logits = tf.reshape(distribution.logits, [-1, shape[-1]])
                for logit in logits:
                    dist = CategoricalDistribution(logits.shape[-1])
                    dist.set_param(tf.stop_gradient(tf.expand_dims(logit, 0)))
                    return_list.append(dist)
            elif isinstance(distribution, DiagGaussianDistribution):
                shape = distribution.mu.shape
                means = tf.reshape(distribution.mu, [-1, shape[-1]])
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
                logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
                action_dim = logits.shape[-1]
                dist = CategoricalDistribution(action_dim)
                dist.set_param(tf.stop_gradient(logits))
                return dist
            elif isinstance(distribution_list[0], DiagGaussianDistribution):
                shape = distribution_list.shape
                distribution_list = distribution_list.reshape([-1])
                mu = tf.concat([dist.mu for dist in distribution_list], axis=0)
                std = tf.concat([dist.std for dist in distribution_list], axis=0)
                action_dim = distribution_list[0].mu.shape[-1]
                dist = DiagGaussianDistribution(action_dim)
                mu = tf.reshape(mu, shape + (action_dim,))
                std = tf.reshape(std, shape + (action_dim,))
                dist.set_param(mu, std)
                return dist
            elif isinstance(distribution_list[0, 0], CategoricalDistribution):
                shape = distribution_list.shape
                distribution_list = distribution_list.reshape([-1])
                logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
                action_dim = logits.shape[-1]
                dist = CategoricalDistribution(action_dim)
                logits = tf.reshape(logits, shape + (action_dim, ))
                dist.set_param(tf.stop_gradient(logits))
                return dist
            else:
                pass


  .. group-tab:: MindSpore

    .. code-block:: python

        import random
        import mindspore as ms
        import mindspore.nn as nn
        import numpy as np
        from mindspore.ops import ExpandDims
        from .distributions import CategoricalDistribution


        def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
            lr = initial_lr * (1 - step / float(total_steps))
            if lr < end_factor * initial_lr:
                lr = end_factor * initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        def set_seed(seed):
            ms.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


        def get_flat_grad(y: ms.Tensor, model: nn.Cell) -> ms.Tensor:
            grads = ms.ops.GradOperation(y, model.parameters())
            return ms.ops.Concat([grad.reshape(-1) for grad in grads])


        def get_flat_params(model: nn.Cell) -> ms.Tensor:
            params = model.parameters()
            return ms.ops.Concat([param.reshape(-1) for param in params])


        def assign_from_flat_grads(flat_grads: ms.Tensor, model: nn.Cell) -> nn.Cell:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def assign_from_flat_params(flat_params: ms.Tensor, model: nn.Cell) -> nn.Cell:
            prev_ind = 0
            for param in model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size
            return model


        def split_distributions(distribution):
            _unsqueeze = ExpandDims()
            return_list = []
            if isinstance(distribution, CategoricalDistribution):
                shape = distribution.logits.shape
                logits = distribution.logits.view(-1,shape[-1])
                for logit in logits:
                    dist = CategoricalDistribution(logits.shape[-1])
                    dist.set_param(_unsqueeze(logit, 0))
                    return_list.append(dist)
            else:
                raise NotImplementedError
            return np.array(return_list).reshape(shape[:-1])


        def merge_distributions(distribution_list):
            if isinstance(distribution_list[0], CategoricalDistribution):
                logits = ms.ops.concat([dist.logits for dist in distribution_list], 0)
                action_dim = logits.shape[-1]
                dist = CategoricalDistribution(action_dim)
                dist.set_param(logits)
                return dist
            else:
                raise NotImplementedError