import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from typing import Optional, Union, Dict
from torch import Tensor
from torch.nn import Module


class IndependentMixer(Module):
    """
    The independent q-learning. (Independent)
    """

    def forward(self, values_n, states=None):
        return values_n


class VDN_Mixer(Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def forward(self, values_n, states=None):
        return values_n.sum(dim=-1, keepdim=True)


class QMIX_Mixer(Module):
    """
    The QMIX mixer. (Monotonicity)

    Args:
        dim_state (int): The dimension of global state.
        dim_hidden (int): The size of rach hidden layer.
        dim_hypernet_hidden (int): The size of rach hidden layer for hyper network.
        n_agents (int): The number of agents.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 dim_state: Optional[int] = None,
                 dim_hidden: int = 32,
                 dim_hypernet_hidden: int = 32,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QMIX_Mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents)).to(device)
        self.hyper_w_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, self.dim_hidden)).to(device)

        self.hyper_b_1 = nn.Linear(self.dim_state, self.dim_hidden).to(device)
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.dim_state, self.dim_hypernet_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hypernet_hidden, 1)).to(device)

    def forward(self, values_n, states):
        """
        Returns the total Q-values for multi-agent team.

        Parameters:
            values_n: The individual values for agents in team.
            states: The global states.

        Returns:
            q_tot: The total Q-values for the multi-agent team.
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.reshape(-1, 1, self.n_agents)
        # First layer
        w_1 = torch.abs(self.hyper_w_1(states))
        w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
        b_1 = self.hyper_b_1(states)
        b_1 = b_1.view(-1, 1, self.dim_hidden)
        hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = torch.abs(self.hyper_w_2(states))
        w_2 = w_2.view(-1, self.dim_hidden, 1)
        b_2 = self.hyper_b_2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot


class QMIX_FF_Mixer(Module):
    """
    The feedforward mixer without the constraints of monotonicity.
    """

    def __init__(self,
                 dim_state: int = 0,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QMIX_FF_Mixer, self).__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        self.ff_net = nn.Sequential(nn.Linear(self.dim_input, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_hidden, 1)).to(self.device)
        self.ff_net_bias = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(self.dim_hidden, 1)).to(self.device)

    def forward(self, values_n, states=None):
        """
        Returns the feedforward total Q-values.

        Parameters:
            values_n: The individual Q-values.
            states: The global states.
        """
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view([-1, self.n_agents])
        inputs = torch.cat([agent_qs, states], dim=-1).to(self.device)
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view([-1, 1])
        return q_tot


class QTRAN_Base(Module):
    """
    The basic QTRAN module.

    Args:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
        device: Optional[Union[str, int, torch.device]]: The calculating device.
    """
    def __init__(self,
                 dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QTRAN_Base, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max
        self.dim_v_input = self.dim_state

        self.Q_jt = nn.Sequential(nn.Linear(self.dim_q_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1)).to(device)
        self.V_jt = nn.Sequential(nn.Linear(self.dim_v_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1)).to(device)
        self.dim_ae_input = dim_utility_hidden + self.n_actions_max
        self.action_encoding = nn.Sequential(nn.Linear(self.dim_ae_input, self.dim_ae_input),
                                             nn.ReLU(),
                                             nn.Linear(self.dim_ae_input, self.dim_ae_input)).to(device)

    def forward(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """
        Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = torch.cat([hidden_state_inputs, actions_onehot], dim=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        h_state_action_encode = h_state_action_encode.sum(dim=1)  # Sum across agents
        input_q = torch.cat([states, h_state_action_encode], dim=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_Alt(Module):
    """
    The basic QTRAN module.

    Parameters:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
        device: Optional[Union[str, int, torch.device]]: The calculating device.
    """

    def __init__(self,
                 dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QTRAN_Alt, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing
        self.device = device

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max + self.n_agents
        self.dim_v_input = self.dim_state

        self.Q_jt = nn.Sequential(nn.Linear(self.dim_q_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.n_actions_max)).to(device)
        self.V_jt = nn.Sequential(nn.Linear(self.dim_v_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1)).to(device)
        self.dim_ae_input = dim_utility_hidden + self.n_actions_max
        self.action_encoding = nn.Sequential(nn.Linear(self.dim_ae_input, self.dim_ae_input),
                                             nn.ReLU(),
                                             nn.Linear(self.dim_ae_input, self.dim_ae_input)).to(device)

    def forward(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = torch.cat([hidden_state_inputs, actions_onehot], dim=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        bs, dim_h = h_state_action_encode.shape[0], h_state_action_encode.shape[-1]
        agent_ids = torch.eye(self.n_agents, dtype=torch.float32, device=self.device)
        agent_masks = (1 - agent_ids)
        repeat_agent_ids = agent_ids.unsqueeze(0).repeat(bs, 1, 1)
        repeated_agent_masks = agent_masks.unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1, dim_h)
        repeated_h_state_action_encode = h_state_action_encode.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        h_state_action_encode = repeated_h_state_action_encode * repeated_agent_masks
        h_state_action_encode = h_state_action_encode.sum(dim=2)  # Sum across other agents

        repeated_states = states.unsqueeze(1).repeat(1, self.n_agents, 1)
        input_q = torch.cat([repeated_states, h_state_action_encode, repeat_agent_ids], dim=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt

