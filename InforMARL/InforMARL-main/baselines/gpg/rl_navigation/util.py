import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_devices(args):
    if args.cuda and torch.cuda.is_available():
        print("_" * 50)
        print("Choose to use GPU")
        print("_" * 50)
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("_" * 50)
        print("Choose to use CPU")
        print("_" * 50)
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    return device


def get_weights(policy):
    w = list(policy.parameters())
    w = np.asarray(w)

    weights = {}

    for name, param in policy.named_parameters():
        if param.requires_grad:
            # print (name, param.data)
            weights["%s" % name] = param.data
            # print (name)
    return weights


def set_weights(new_policy, old_policy):
    w = get_weights(old_policy)
    w2 = get_weights(new_policy)


def check(input) -> Tensor:
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_grad_norm(it) -> float:
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d) -> float:
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape
