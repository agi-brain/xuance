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
        dist.set_param(logits=logits.detach())
        return dist
    else:
        pass
