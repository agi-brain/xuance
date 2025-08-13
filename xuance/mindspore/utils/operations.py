import random
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore.ops import ExpandDims, Concat, clip_by_value
from .distributions import CategoricalDistribution, DiagGaussianDistribution


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


def clip_grads(grads, low, high):
    new_grads = ()
    for grad in grads:
        t = clip_by_value(grad, low, high)
        new_grads = new_grads + (t, )
    return new_grads
