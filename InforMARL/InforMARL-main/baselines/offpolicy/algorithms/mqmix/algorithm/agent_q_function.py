import torch
import torch.nn as nn

from baselines.offpolicy.utils.util import to_torch
from baselines.offpolicy.algorithms.utils.mlp import MLPBase
from baselines.offpolicy.algorithms.utils.act import ACTLayer


class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """

    def __init__(self, args, input_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mlp = MLPBase(args, input_dim)
        self.q = ACTLayer(
            act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain
        )
        self.to(device)

    def forward(self, x):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        q_value = self.q(x)

        return q_value
