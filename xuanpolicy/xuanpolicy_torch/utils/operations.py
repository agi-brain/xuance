import torch
import torch.nn as nn
import numpy as np
from .distributions import CategoricalDistribution, DiagGaussianDistribution


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
