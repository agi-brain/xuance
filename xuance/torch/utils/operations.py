import os
import random
import torch
import numpy as np
from torch import nn, Tensor
from torch.distributed import init_process_group
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from xuance.common import Dict, Any, Optional


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


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def sym_log(x: Tensor) -> Tensor:
    """Computes the symmetric logarithm of a tensor.

    This function applies a logarithm transformation while preserving the sign of the input.
    The operation is defined as: sign(x) * log(1 + |x|).
    Ref: https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py

    Args:
        x: Input tensor of any shape.

    Returns:
        A tensor of the same shape as input, with symmetric logarithm applied element-wise.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def sym_exp(x: Tensor) -> Tensor:
    """Computes the symmetric exponential of a tensor.

    This function applies an exponential transformation while preserving the sign of the input.
    The operation is defined as: sign(x) * (exp(|x|) - 1).

    Args:
        x: Input tensor of any shape.

    Returns:
        A tensor of the same shape as input, with symmetric exponential applied element-wise.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot_encoder(tensor: Tensor, support_range: int = 300, num_buckets: Optional[int] = None) -> Tensor:
    """Encode a tensor representing a floating point number `x` as a tensor with all zeros except for two entries in the
    indexes of the two buckets closer to `x` in the support of the distribution.
    Check https://arxiv.org/pdf/2301.04104v1.pdf equation 9 for more details.

    Args:
        tensor (Tensor): tensor to encode of shape (..., batch_size, 1)
        support_range (int): range of the support of the distribution, going from -support_range to support_range
        num_buckets (int): number of buckets in the support of the distribution

    Returns:
        Tensor: tensor of shape (..., batch_size, support_size)
    """
    if tensor.shape == torch.Size([]):
        tensor = tensor.unsqueeze(0)
    if num_buckets is None:
        num_buckets = support_range * 2 + 1
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    tensor = tensor.clip(-support_range, support_range)
    buckets = torch.linspace(-support_range, support_range, num_buckets, device=tensor.device)
    bucket_size = buckets[1] - buckets[0] if len(buckets) > 1 else 1.0

    right_idxs = torch.bucketize(tensor, buckets)
    left_idxs = (right_idxs - 1).clip(min=0)

    two_hot = torch.zeros(tensor.shape[:-1] + (num_buckets,), device=tensor.device)
    left_value = torch.abs(buckets[right_idxs] - tensor) / bucket_size
    right_value = 1 - left_value
    two_hot.scatter_add_(-1, left_idxs, left_value)
    two_hot.scatter_add_(-1, right_idxs, right_value)

    return two_hot


def two_hot_decoder(tensor: torch.Tensor, support_range: int) -> torch.Tensor:
    """Decode a tensor representing a two-hot vector as a tensor of floating point numbers.

    Args:
        tensor (Tensor): tensor to decode of shape (..., batch_size, support_size)
        support_range (int): range of the support of the values, going from -support_range to support_range

    Returns:
        Tensor: tensor of shape (..., batch_size, 1)
    """
    num_buckets = tensor.shape[-1]
    if num_buckets % 2 == 0:
        raise ValueError("support_size must be odd")
    support = torch.linspace(-support_range, support_range, num_buckets).to(tensor.device)
    return torch.sum(tensor * support, dim=-1, keepdim=True)


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


class dotdict(dict):
    """
    A dictionary supporting dot notation.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def as_dict(self) -> Dict[str, Any]:
        _copy = dict(self)
        for k, v in _copy.items():
            if isinstance(v, dotdict):
                _copy[k] = v.as_dict()
        return _copy

