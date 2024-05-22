import torch
import torch.nn as nn
from baselines.offpolicy.utils.util import init, to_torch
from baselines.offpolicy.algorithms.utils.rnn import RNNBase
from baselines.offpolicy.algorithms.utils.act import ACTLayer


class R_MADDPG_Actor(nn.Module):
    """
    Actor network class for recurrent MADDPG/MATD3. Outputs actions given observations + rnn state.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_dim: (int) dimension of the observation vector.
    :param act_dim: (int) dimension of the action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param take_prev_action: (bool) whether the previous action should be part of the network input.
    """

    def __init__(self, args, obs_dim, act_dim, device, take_prev_action=False):
        super(R_MADDPG_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.take_prev_act = take_prev_action
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = (obs_dim + act_dim) if take_prev_action else obs_dim

        # map observation input into input for rnn
        self.rnn = RNNBase(args, input_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, prev_acts, rnn_states):
        """
        Compute actions using the needed information.
        :param obs: (np.ndarray) Observations with which to compute actions.
        :param prev_actions: (np.ndarray) Optionally use previous action to  compute actions.
        :param rnn_states: (np.ndarray / torch.Tensor) RNN state to use to compute actions
        """
        # make sure input is a torch tensor
        obs = to_torch(obs).to(**self.tpdv)
        rnn_states = to_torch(rnn_states).to(**self.tpdv)
        if prev_acts is not None:
            prev_acts = to_torch(prev_acts).to(**self.tpdv)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            if self.take_prev_act:
                prev_acts = prev_acts[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)
        if len(rnn_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_states = rnn_states[None]

        inp = torch.cat((obs, prev_acts), dim=-1) if self.take_prev_act else obs

        rnn_outs, h_final = self.rnn(inp, rnn_states)
        # pass outputs through linear layer
        act_outs = self.act(rnn_outs, no_sequence)

        return act_outs, h_final


class R_MADDPG_Critic(nn.Module):
    """
    Critic network class for recurrent MADDPG/MATD3. Outputs actions given observations + rnn state.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """

    def __init__(self, args, central_obs_dim, central_act_dim, device, num_q_outs=1):
        super(R_MADDPG_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_q_outs = num_q_outs

        input_dim = central_obs_dim + central_act_dim

        self.rnn = RNNBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_outs = nn.ModuleList(
            [init_(nn.Linear(self.hidden_size, 1)) for _ in range(self.num_q_outs)]
        )

        self.to(device)

    def forward(self, central_obs, central_act, rnn_states):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.
        :param rnn_states: (np.ndarray / torch.Tensor) RNN state to use to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        # ensure inputs are torch tensors
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)
        rnn_states = to_torch(rnn_states).to(**self.tpdv)

        no_sequence = False
        if len(central_obs.shape) == 2 and len(central_act.shape) == 2:
            # no sequence, so add a time dimension of len 0
            no_sequence = True
            central_obs, central_act = central_obs[None], central_act[None]

        if len(rnn_states.shape) == 2:
            # also add a first dimension to the rnn hidden states
            rnn_states = rnn_states[None]

        inp = torch.cat([central_obs, central_act], dim=2)

        rnn_outs, h_final = self.rnn(inp, rnn_states)
        q_values = [q_out(rnn_outs) for q_out in self.q_outs]

        if no_sequence:
            # remove the time dimension
            q_values = [q[0, :, :] for q in q_values]

        return q_values, h_final
