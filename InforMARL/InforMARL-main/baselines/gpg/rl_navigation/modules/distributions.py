import copy
import numpy as np
import torch
import torch.nn as nn

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""


def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


#
# Standardize distribution interfaces
#


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

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


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    """
    Categorical Distribution for discrete action space modified to
    renormalise probs with available_actions.
    Has a linear layer followed by renormalisation of the obtained logits
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        use_orthogonal: bool = True,
        gain: float = 0.01,
    ):
        """
        Params
        num_inputs: int
            The input dimension for the linear layer
        num_outputs: int
            The output dimension for the linear layer
        use_orthogonal: bool
            Whether we want to use orthogonal weight init or Xavier uniform
        gain: float
            The gain for weight init
        """
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.tensor, available_actions=None):
        x = self.linear(x)
        # supress the logits at all non-available actions
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian Distribution for continuous action space modified to
    renormalise probs with available_actions.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        use_orthogonal: bool = True,
        gain: float = 0.01,
    ):
        """
        Params
        num_inputs: int
            The input dimension for the linear layer
        num_outputs: int
            The output dimension for the linear layer
        use_orthogonal: bool
            Whether we want to use orthogonal weight init or Xavier uniform
        gain: float
            The gain for weight init
        """
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.tensor):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    """
    Bernoulli Distribution for discrete action space modified to
    renormalise probs with available_actions.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        use_orthogonal: bool = True,
        gain: float = 0.01,
    ):
        """
        Params
        num_inputs: int
            The input dimension for the linear layer
        num_outputs: int
            The output dimension for the linear layer
        use_orthogonal: bool
            Whether we want to use orthogonal weight init or Xavier uniform
        gain: float
            The gain for weight init
        """
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    def __init__(self, bias: torch.tensor):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.tensor):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
