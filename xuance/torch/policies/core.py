import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional, Callable, Union
from xuance.torch import Tensor, Module
from xuance.torch.utils import ModuleType, mlp_block, gru_block, lstm_block
from xuance.torch.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution


class BasicQhead(Module):
    """
    A base class to build Q network and calculate the Q values.

    Args:
        state_dim (int): The input state dimension.
        n_actions (int): The number of discrete actions.
        hidden_sizes: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], n_actions, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class DuelQhead(Module):
    """
    A base class to build Q network and calculate the dueling Q values.

    Args:
        state_dim (int): The input state dimension.
        n_actions (int): The number of discrete actions.
        hidden_sizes: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(DuelQhead, self).__init__()
        v_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
            v_layers.extend(v_mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        a_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize, device)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], n_actions, None, None, None, device)[0])
        self.a_model = nn.Sequential(*a_layers)
        self.v_model = nn.Sequential(*v_layers)

    def forward(self, x: Tensor):
        """
        Returns the dueling Q-values.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            q: The dueling Q-values.
        """
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - a.mean(dim=-1).unsqueeze(dim=-1))
        return q


class C51Qhead(Module):
    """
    A base class to build Q network and calculate the distributional Q values.

    Args:
        state_dim (int): The input state dimension.
        n_actions (int): The number of discrete actions.
        atom_num (int): The number of atoms.
        hidden_sizes: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(C51Qhead, self).__init__()
        self.n_actions = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the discrete action distributions.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            dist_probs: The probability distribution of the discrete actions.
        """
        dist_logits = self.model(x).view(-1, self.n_actions, self.atom_num)
        dist_probs = F.softmax(dist_logits, dim=-1)
        return dist_probs


class QRDQNhead(Module):
    """
    A base class to build Q networks for QRDQN policy.

    Args:
        state_dim (int): The input state dimension.
        n_actions (int): The number of discrete actions.
        atom_num (int): The number of atoms.
        hidden_sizes: List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QRDQNhead, self).__init__()
        self.n_actions = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the quantiles of the distribution.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            quantiles: The quantiles of the action distribution.
        """
        quantiles = self.model(x).view(-1, self.n_actions, self.atom_num)
        return quantiles


class BasicRecurrent(Module):
    """Build recurrent  neural network to calculate Q values."""

    def __init__(self, **kwargs):
        super(BasicRecurrent, self).__init__()
        self.lstm = False
        if kwargs["rnn"] == "GRU":
            output, _ = gru_block(kwargs["input_dim"],
                                  kwargs["recurrent_hidden_size"],
                                  kwargs["recurrent_layer_N"],
                                  kwargs["dropout"],
                                  kwargs["initialize"],
                                  kwargs["device"])
        elif kwargs["rnn"] == "LSTM":
            self.lstm = True
            output, _ = lstm_block(kwargs["input_dim"],
                                   kwargs["recurrent_hidden_size"],
                                   kwargs["recurrent_layer_N"],
                                   kwargs["dropout"],
                                   kwargs["initialize"],
                                   kwargs["device"])
        else:
            raise "Unknown recurrent module!"
        self.rnn_layer = output
        fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None, kwargs["device"])[
            0]
        self.model = nn.Sequential(*fc_layer)

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None):
        """Returns the rnn hidden and Q-values via RNN networks."""
        self.rnn_layer.flatten_parameters()
        if self.lstm:
            output, (hn, cn) = self.rnn_layer(x, (h, c))
            return hn, cn, self.model(output)
        else:
            output, hn = self.rnn_layer(x, h)
            return hn, self.model(output)


class ActorNet(nn.Module):
    """
    The actor network for deterministic policy, which outputs activated continuous actions directly.
    
    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the output of the actor.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class CategoricalActorNet(Module):
    """
    The actor network for categorical policy, which outputs a distribution over all discrete actions.

    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CategoricalActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)
        self.dist = CategoricalDistribution(action_dim)

    def forward(self, x: Tensor, avail_actions: Optional[Tensor] = None):
        """
        Returns the stochastic distribution over all discrete actions.
        Parameters:
            x (Tensor): The input tensor.
            avail_actions (Optional[Tensor]): The actions mask values when use actions mask, default is None.

        Returns:
            self.dist: CategoricalDistribution(action_dim), a distribution over all discrete actions.
        """
        logits = self.model(x)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        self.dist.set_param(logits=logits)
        return self.dist


class CategoricalActorNet_SAC(CategoricalActorNet):
    """
    The actor network for categorical policy in SAC-DIS, which outputs a distribution over all discrete actions.

    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CategoricalActorNet_SAC, self).__init__(state_dim, action_dim, hidden_sizes,
                                                      normalize, initialize, activation, device)
        self.output = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, avail_actions: Optional[Tensor] = None):
        """
        Returns the stochastic distribution over all discrete actions.
        Parameters:
            x (Tensor): The input tensor.
            avail_actions (Optional[Tensor]): The actions mask values when use actions mask, default is None.

        Returns:
            self.dist: CategoricalDistribution(action_dim), a distribution over all discrete actions.
        """
        logits = self.model(x)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        probs = self.output(logits)
        self.dist.set_param(probs=probs)
        return self.dist


class GaussianActorNet(Module):
    """
    The actor network for Gaussian policy, which outputs a distribution over the continuous action space.

    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(GaussianActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize, device)[0])
        self.mu = nn.Sequential(*layers)
        self.logstd = nn.Parameter(-torch.ones((action_dim,), device=device))
        self.dist = DiagGaussianDistribution(action_dim)

    def forward(self, x: Tensor):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        self.dist.set_param(self.mu(x), self.logstd.exp())
        return self.dist


class CriticNet(Module):
    """
    The critic network that outputs the evaluated values for states (State-Value) or state-action pairs (Q-value).

    Args:
        input_dim (int): The input dimension (dim_state or dim_state + dim_action).
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (input_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class GaussianActorNet_SAC(Module):
    """
    The actor network for Gaussian policy in SAC, which outputs a distribution over the continuous action space.

    Args:
        state_dim (int): The input state dimension.
        action_dim (int): The dimension of continuous action space.
        hidden_sizes (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(GaussianActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.out_log_std = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action, device)

    def forward(self, x: Tensor):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        output = self.output(x)
        mu = self.out_mu(output)
        log_std = torch.clamp(self.out_log_std(output), -20, 2)
        std = log_std.exp()
        self.dist.set_param(mu, std)
        return self.dist


class VDN_mixer(nn.Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def __init__(self):
        super(VDN_mixer, self).__init__()

    def forward(self, values_n, states=None):
        return values_n.sum(dim=1)


class QMIX_mixer(nn.Module):
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
        super(QMIX_mixer, self).__init__()
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


class QMIX_FF_mixer(nn.Module):
    """
    The feedforward mixer without the constraints of monotonicity.
    """

    def __init__(self,
                 dim_state: int = 0,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(QMIX_FF_mixer, self).__init__()
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

    def forward(self, values_n, states):
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


class QTRAN_base(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

        self.Q_jt = nn.Sequential(nn.Linear(self.dim_q_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1))
        self.V_jt = nn.Sequential(nn.Linear(self.dim_v_input, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, self.dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(self.dim_hidden, 1))

    def forward(self, hidden_states_n, actions_n):
        input_q = torch.cat([hidden_states_n, actions_n], dim=-1).view([-1, self.dim_q_input])
        input_v = hidden_states_n.view([-1, self.dim_v_input])
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = q_selected_values.unsqueeze(dim=1).repeat(1, self.n_agents, 1, self.dim_action)
        counterfactual_values_n = q_repeat
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
        return counterfactual_values_n.sum(dim=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = actions_n.unsqueeze(dim=2).repeat(1, 1, self.dim_action, 1)
        action_self_all = torch.eye(self.dim_action).unsqueeze(0)
        action_counterfactual_n = action_repeat.unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1,
                                                                        1)  # batch * N * N * dim_a * dim_a
        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = action_counterfactual_n[:, :, agent, a, :]
                q, _ = self.forward(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(torch.cat(q_actions, dim=-1).unsqueeze(dim=1))
        return torch.cat(q_n, dim=1)
