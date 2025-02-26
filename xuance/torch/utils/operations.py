import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.distributed import init_process_group
from .distributions import CategoricalDistribution, DiagGaussianDistribution


def init_distributed_mode(master_port: str = None):
    """Initializes the distributed training environment.

    This function sets up the necessary environment variables for distributed training,
    configures the CUDA device for each process, and initializes the process group
    for communication between GPUs using the NCCL backend.

    Args:
        master_port (str, optional): The port number for the master process.
            If not provided, the default value "12355" is used.
    """
    rank = os.environ["LOCAL_RANK"]
    os.environ["MASTER_ADDR"] = "localhost"  # The IP address of the machine that is running the rank 0 process.
    os.environ["MASTER_PORT"] = "12355" if master_port is None else master_port
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend='nccl')
    if rank == 0:
        print("The distributed process group is initialized.")


def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
    """Updates the learning rate using a linear decay schedule.

    This function adjusts the learning rate linearly based on the current step.
    The learning rate starts from `initial_lr` and decreases linearly until it
    reaches a minimum value determined by `end_factor * initial_lr`.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
        step (int): The current training step.
        total_steps (int): The total number of training steps.
        initial_lr (float): The initial learning rate at step 0.
        end_factor (float): The minimum learning rate factor. The final learning rate
            is at least `end_factor * initial_lr`.

    """
    lr = initial_lr * (1 - step / float(total_steps))
    if lr < end_factor * initial_lr:
        lr = end_factor * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_seed(seed):
    """Sets the random seed for reproducibility across different libraries.

    This function ensures that random number generation in PyTorch (CPU & GPU),
    NumPy, and Python's built-in `random` module produces consistent results
    across runs, improving experiment reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_flat_grad(y: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Computes and flattens the gradients of a given loss tensor with respect to model parameters.

    Args:
        y (torch.Tensor): The scalar loss tensor whose gradients will be computed.
        model (torch.nn.Module): The neural network model whose parameters' gradients
            need to be extracted.

    Returns:
        torch.Tensor: A 1D tensor containing all the gradients of the model parameters,
        concatenated into a single vector.
    """
    grads = torch.autograd.grad(y, model.parameters())
    return torch.cat([grad.reshape(-1) for grad in grads])


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flattens and concatenates all parameters of a given model into a 1D tensor.

    Args:
        model (torch.nn.Module): The neural network model whose parameters need to be flattened.

    Returns:
        torch.Tensor: A 1D tensor containing all model parameters concatenated together.
    """
    params = model.parameters()
    return torch.cat([param.reshape(-1) for param in params])


def assign_from_flat_grads(flat_grads: torch.Tensor, model: nn.Module) -> nn.Module:
    """Assigns gradients from a flat tensor to the corresponding model parameters.

    This function takes a flattened gradient tensor and reshapes it to match the model's
    parameter shapes before assigning the values to the respective parameter gradients.

    Args:
        flat_grads (torch.Tensor): A 1D tensor containing all model gradients in a flattened form.
        model (torch.nn.Module): The model whose parameters will be updated with the given gradients.

    Returns:
        torch.nn.Module: The model with updated parameter gradients.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


def assign_from_flat_params(flat_params: torch.Tensor, model: nn.Module) -> nn.Module:
    """Assigns parameter values from a flat tensor to the corresponding model parameters.

    This function takes a flattened parameter tensor and reshapes it to match the model's
    parameter shapes before assigning the values to the respective model parameters.

    Args:
        flat_params (torch.Tensor): A 1D tensor containing all model parameters in a flattened form.
        model (torch.nn.Module): The model whose parameters will be updated with the given values.

    Returns:
        torch.nn.Module: The model with updated parameters.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


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
