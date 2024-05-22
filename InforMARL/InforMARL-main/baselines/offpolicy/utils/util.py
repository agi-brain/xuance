import copy
import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input


def to_numpy(x):
    return x.detach().cpu().numpy()


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [
            int(x)
            for x in np.floor(
                np.multiply((self.high - self.low + 1.0), random_array) + self.low
            )
        ]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(
            self.high, other.high
        )


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))

    pass


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def onehot_from_logits(logits, avail_logits=None, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    logits = to_torch(logits)

    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits)
        logits[avail_logits == 0] = -1e10
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(
    logits, avail_logits, temperature, device=torch.device("cpu")
):
    """Draw a sample from the Gumbel-Softmax distribution"""
    if str(device) == "cpu":
        y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    else:
        y = (
            logits.cpu() + sample_gumbel(logits.shape, tens_type=type(logits.data))
        ).cuda()

    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits).to(device)
        y[avail_logits == 0] = -1e10
    return F.softmax(y / temperature, dim=dim)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(
    logits, avail_logits=None, temperature=1.0, hard=False, device=torch.device("cpu")
):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, avail_logits, temperature, device)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def gaussian_noise(shape, std):
    return torch.empty(shape).normal_(mean=0, std=std)


def get_obs_shape(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError

    return obs_shape


def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif "MultiDiscrete" in space.__class__.__name__:
        return (space.high - space.low) + 1
    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim


def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum(
        [get_dim_from_space(space) for space in observation_dict.values()]
    )
    combined_act_dim = 0
    for space in action_dict.values():
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim + combined_act_dim


def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim


def is_discrete(space):
    if isinstance(space, Discrete) or "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def is_multidiscrete(space):
    if "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def make_onehot(int_action, action_dim, seq_len=None):
    if type(int_action) == torch.Tensor:
        int_action = int_action.cpu().numpy()
    if not seq_len:
        return np.eye(action_dim)[int_action]
    if seq_len:
        onehot_actions = []
        for i in range(seq_len):
            onehot_action = np.eye(action_dim)[int_action[i]]
            onehot_actions.append(onehot_action)
        return np.stack(onehot_actions)


def avail_choose(x, avail_x=None):
    x = to_torch(x)
    if avail_x is not None:
        avail_x = to_torch(avail_x)
        x[avail_x == 0] = -1e10
    return x  # FixedCategorical(logits=x)


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


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
