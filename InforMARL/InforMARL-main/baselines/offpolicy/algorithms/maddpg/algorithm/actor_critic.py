import torch
import torch.nn as nn
from baselines.offpolicy.utils.util import init, to_torch
from baselines.offpolicy.algorithms.utils.mlp import MLPBase
from baselines.offpolicy.algorithms.utils.act import ACTLayer


class MADDPG_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device):
        """
        Actor network class for MADDPG/MATD3. Outputs actions given observations.
        :param args: (argparse.Namespace) arguments containing relevant model information.
        :param obs_dim: (int) dimension of the observation vector.
        :param act_dim: (int) dimension of the action vector.
        :param device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(MADDPG_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, x):
        """
        Compute actions using the needed information.
        :param x: (np.ndarray) Observations with which to compute actions.
        """
        x = to_torch(x).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        action = self.act(x)

        return action


class MADDPG_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """

    def __init__(self, args, central_obs_dim, central_act_dim, device, num_q_outs=1):
        super(MADDPG_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = central_obs_dim + central_act_dim

        self.mlp = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_outs = [init_(nn.Linear(self.hidden_size, 1)) for _ in range(num_q_outs)]

        self.to(device)

    def forward(self, central_obs, central_act):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)

        x = torch.cat([central_obs, central_act], dim=1)

        x = self.mlp(x)
        q_values = [q_out(x) for q_out in self.q_outs]

        return q_values
