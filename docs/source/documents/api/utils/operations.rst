Operations
===========================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:function::
  xuance.torch.utils.operations.update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor)

  xxxxxx.

  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param step: xxxxxx.
  :type step: xxxxxx
  :param total_steps: xxxxxx.
  :type total_steps: xxxxxx
  :param initial_lr: xxxxxx.
  :type initial_lr: xxxxxx
  :param end_factor: xxxxxx.
  :type end_factor: xxxxxx

.. py:function::
  xuance.torch.utils.operations.set_seed(seed)

  xxxxxx.

  :param seed: xxxxxx.
  :type seed: xxxxxx

.. py:function::
  xuance.torch.utils.operations.get_flat_grad(y, model)

  xxxxxx.

  :param y: xxxxxx.
  :type y: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Tensor

.. py:function::
  xuance.torch.utils.operations.get_flat_params(model)

  xxxxxx.

  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Tensor

.. py:function::
  xuance.torch.utils.operations.assign_from_flat_grads(flat_grads, model)

  xxxxxx.

  :param flat_grads: xxxxxx.
  :type flat_grads: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Module

.. py:function::
  xuance.torch.utils.operations.assign_from_flat_params(flat_grads, model)

  xxxxxx.

  :param flat_grads: xxxxxx.
  :type flat_grads: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Module

.. py:function::
  xuance.torch.utils.operations.split_distributions(distribution)

  xxxxxx.

  :param distribution: xxxxxx.
  :type distribution: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.utils.operations.merge_distributions(distribution_list)

  xxxxxx.

  :param distribution_list: xxxxxx.
  :type distribution_list: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:function::
  xuance.mindspore.utils.operations.update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor)

  xxxxxx.

  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param step: xxxxxx.
  :type step: xxxxxx
  :param total_steps: xxxxxx.
  :type total_steps: xxxxxx
  :param initial_lr: xxxxxx.
  :type initial_lr: xxxxxx
  :param end_factor: xxxxxx.
  :type end_factor: xxxxxx

.. py:function::
  xuance.mindspore.utils.operations.set_seed(seed)

  xxxxxx.

  :param seed: xxxxxx.
  :type seed: xxxxxx

.. py:function::
  xuance.mindspore.utils.operations.get_flat_grad(y, model)

  xxxxxx.

  :param y: xxxxxx.
  :type y: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Tensor

.. py:function::
  xuance.mindspore.utils.operations.get_flat_params(model)

  xxxxxx.

  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Tensor

.. py:function::
  xuance.mindspore.utils.operations.assign_from_flat_grads(flat_grads, model)

  xxxxxx.

  :param flat_grads: xxxxxx.
  :type flat_grads: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Module

.. py:function::
  xuance.mindspore.utils.operations.assign_from_flat_params(flat_grads, model)

  xxxxxx.

  :param flat_grads: xxxxxx.
  :type flat_grads: xxxxxx
  :param model: xxxxxx.
  :type model: xxxxxx
  :return: xxxxxx.
  :rtype: Module

.. py:function::
  xuance.mindspore.utils.operations.split_distributions(distribution)

  xxxxxx.

  :param distribution: xxxxxx.
  :type distribution: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.utils.operations.merge_distributions(distribution_list)

  xxxxxx.

  :param distribution_list: xxxxxx.
  :type distribution_list: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

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