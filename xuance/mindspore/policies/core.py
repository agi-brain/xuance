import mindspore.nn as nn
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Dict
from xuance.mindspore import Tensor, Module, ms, ops
from xuance.mindspore.utils import ModuleType, mlp_block, gru_block, lstm_block
from xuance.mindspore.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution


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
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQhead, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: Tensor):
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
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(DuelQhead, self).__init__()
        v_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            v_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
            v_layers.extend(v_mlp)
        v_layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])

        a_layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            a_mlp, input_shape = mlp_block(input_shape[0], h // 2, normalize, activation, initialize)
            a_layers.extend(a_mlp)
        a_layers.extend(mlp_block(input_shape[0], n_actions, None, None, None)[0])

        self.a_model = nn.SequentialCell(*a_layers)
        self.v_model = nn.SequentialCell(*v_layers)
        self.reduce_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x: Tensor):
        """
        Returns the dueling Q-values.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            q: The dueling Q-values.
        """
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - self.reduce_mean(a, axis=-1))
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
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(C51Qhead, self).__init__()
        self.n_actions = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)
        self._softmax = ms.ops.Softmax(axis=-1)

    def construct(self, x: Tensor):
        """
        Returns the discrete action distributions.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            dist_probs: The probability distribution of the discrete actions.
        """
        dist_logits = self.model(x).reshape([-1, self.n_actions, self.atom_num])
        dist_probs = self._softmax(dist_logits)
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
    """

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(QRDQNhead, self).__init__()
        self.n_actions = n_actions
        self.atom_num = int(atom_num)
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: Tensor):
        """
        Returns the quantiles of the distribution.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            quantiles: The quantiles of the action distribution.
        """
        quantiles = self.model(x).reshape([-1, self.n_actions, self.atom_num])
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
                                  kwargs["initialize"])
        elif kwargs["rnn"] == "LSTM":
            self.lstm = True
            output, _ = lstm_block(kwargs["input_dim"],
                                   kwargs["recurrent_hidden_size"],
                                   kwargs["recurrent_layer_N"],
                                   kwargs["dropout"],
                                   kwargs["initialize"])
        else:
            raise "Unknown recurrent module!"
        self.rnn_layer = output
        fc_layer = mlp_block(kwargs["recurrent_hidden_size"], kwargs["action_dim"], None, None, None)[0]
        self.model = nn.SequentialCell(*fc_layer)

    def construct(self, x: Tensor, h: Tensor, c: Tensor = None):
        """Returns the rnn hidden and Q-values via RNN networks."""
        # self.rnn_layer.flatten_parameters()
        if self.lstm:
            output, (hn, cn) = self.rnn_layer(Tensor(x), (Tensor(h), Tensor(c)))
            return hn, cn, self.model(output)
        else:
            output, hn = self.rnn_layer(Tensor(x), Tensor(h))
            return hn, self.model(output)


class ActorNet(Module):
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
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: Tensor, avail_actions: Optional[Tensor] = None):
        """
        Returns the output of the actor.
        Parameters:
            x (Tensor): The input tensor.
            avail_actions (Optional[Tensor]): The actions mask values when use actions mask, default is None.
        """
        logits = self.model(x)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        return logits


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
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CategoricalActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)
        self.dist = CategoricalDistribution(action_dim)

    def construct(self, x: Tensor, avail_actions: Optional[Tensor] = None):
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
        return logits

    def distribution(self, logits: Tensor):
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
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CategoricalActorNet_SAC, self).__init__(state_dim, action_dim, hidden_sizes,
                                                      normalize, initialize, activation)
        self.output = nn.Softmax(axis=-1)

    def construct(self, x: Tensor, avail_actions: Optional[Tensor] = None):
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
        return logits

    def distribution(self, logits: Tensor):
        self.dist.set_param(logits=logits)
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
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(GaussianActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize)[0])
        self.mu = nn.SequentialCell(*layers)
        self._ones = ms.ops.Ones()
        self.logstd = ms.Parameter(-self._ones((action_dim,), ms.float32))
        self.dist = DiagGaussianDistribution(action_dim)

    def construct(self, x: Tensor):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        mu_ = self.mu(x)
        std_ = ops.exp(self.logstd)
        return mu_, std_

    def distribution(self, mu: Tensor, std: Tensor):
        self.dist.set_param(mu=mu, std=std)
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
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (input_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: Tensor):
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
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None):
        super(GaussianActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        self.output = nn.SequentialCell(*layers)
        self.out_mu = nn.Dense(hidden_sizes[-1], action_dim)
        self.out_log_std = nn.Dense(hidden_sizes[-1], action_dim)
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action)

    def construct(self, x: Tensor):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        output = self.output(x)
        mu_ = self.out_mu(output)
        log_std = ops.clip_by_value(self.out_log_std(output), -20, 2)
        std_ = ops.exp(log_std)
        return mu_, std_

    def distribution(self, mu: Tensor, std: Tensor):
        self.dist.set_param(mu=mu, std=std)
        return self.dist


class VDN_mixer(Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def __init__(self):
        super(VDN_mixer, self).__init__()

    def construct(self, values_n, states=None):
        return ops.reduce_sum(values_n, axis=1)


class QMIX_mixer(Module):
    """
    The QMIX mixer. (Monotonicity)

    Args:
        dim_state (int): The dimension of global state.
        dim_hidden (int): The size of rach hidden layer.
        dim_hypernet_hidden (int): The size of rach hidden layer for hyper network.
        n_agents (int): The number of agents.
    """

    def __init__(self, dim_state: Optional[int] = None,
                 dim_hidden: int = 32,
                 dim_hypernet_hidden: int = 32,
                 n_agents: int = 1):
        super(QMIX_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, self.dim_hidden * self.n_agents))
        self.hyper_w_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, self.dim_hidden))

        self.hyper_b_1 = nn.Dense(self.dim_state, self.dim_hidden)
        self.hyper_b_2 = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hypernet_hidden),
                                           nn.ReLU(),
                                           nn.Dense(self.dim_hypernet_hidden, 1))
        self._abs = ms.ops.Abs()
        self._elu = ms.ops.Elu()

    def construct(self, values_n, states):
        """
        Returns the total Q-values for multi-agent team.

        Parameters:
            values_n: The individual values for agents in team.
            states: The global states.

        Returns:
            q_tot: The total Q-values for the multi-agent team.
        """

        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, 1, self.n_agents)
        # First layer
        w_1 = self._abs(self.hyper_w_1(states))
        w_1 = w_1.view(-1, self.n_agents, self.dim_hidden)
        b_1 = self.hyper_b_1(states)
        b_1 = b_1.view(-1, 1, self.dim_hidden)
        hidden = self._elu(ms.ops.matmul(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = self._abs(self.hyper_w_2(states))
        w_2 = w_2.view(-1, self.dim_hidden, 1)
        b_2 = self.hyper_b_2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = ms.ops.matmul(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot


class QMIX_FF_mixer(Module):
    """
    The feedforward mixer without the constraints of monotonicity.
    """

    def __init__(self, dim_state: int = 0,
                 dim_hidden: int = 32,
                 n_agents: int = 1):
        super(QMIX_FF_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_input = self.n_agents + self.dim_state
        self.ff_net = nn.SequentialCell(nn.Dense(self.dim_input, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, self.dim_hidden),
                                        nn.ReLU(),
                                        nn.Dense(self.dim_hidden, 1))
        self.ff_net_bias = nn.SequentialCell(nn.Dense(self.dim_state, self.dim_hidden),
                                             nn.ReLU(),
                                             nn.Dense(self.dim_hidden, 1))
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, values_n, states):
        """
        Returns the feedforward total Q-values.

        Parameters:
            values_n: The individual Q-values.
            states: The global states.
        """

        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, self.n_agents)
        inputs = self._concat([agent_qs, states])
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view(-1, 1)
        return q_tot


class QTRAN_base(Module):
    """
    The basic QTRAN module.

    Args:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
    """

    def __init__(self, dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = int(self.dim_state + dim_utility_hidden + self.n_actions_max)
        self.dim_v_input = int(self.dim_state)

        self.Q_jt = nn.SequentialCell(nn.Dense(self.dim_q_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self.V_jt = nn.SequentialCell(nn.Dense(self.dim_v_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self.dim_ae_input = int(dim_utility_hidden + self.n_actions_max)
        self.action_encoding = nn.SequentialCell(nn.Dense(self.dim_ae_input, self.dim_ae_input),
                                                 nn.ReLU(),
                                                 nn.Dense(self.dim_ae_input, self.dim_ae_input))

    def construct(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
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
        h_state_action_input = ops.cat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        h_state_action_encode = ops.sum(h_state_action_encode, dim=1)  # Sum across agents
        input_q = ops.cat([states, h_state_action_encode], axis=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(Module):
    """
    The basic QTRAN module.

    Parameters:
        dim_state (int): The dimension of the global state.
        action_space (Dict[str, Discrete]): The action space for all agents.
        dim_hidden (int): The dimension of the hidden layers.
        n_agents (int): The number of agents.
        dim_utility_hidden (int): The dimension of the utility hidden states.
        use_parameter_sharing (bool): Whether to use parameters sharing trick.
    """

    def __init__(self, dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False):
        super(QTRAN_alt, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max + self.n_agents
        self.dim_v_input = self.dim_state

        self.Q_jt = nn.SequentialCell(nn.Dense(self.dim_q_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.n_actions_max))
        self.V_jt = nn.SequentialCell(nn.Dense(self.dim_v_input, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Dense(self.dim_hidden, 1))
        self.dim_ae_input = dim_utility_hidden + self.n_actions_max
        self.action_encoding = nn.SequentialCell(nn.Dense(self.dim_ae_input, self.dim_ae_input),
                                                 nn.ReLU(),
                                                 nn.Dense(self.dim_ae_input, self.dim_ae_input))

    def construct(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor):
        """Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = ops.cat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = self.action_encoding(h_state_action_input).reshape(-1, self.n_agents, self.dim_ae_input)
        bs, dim_h = h_state_action_encode.shape[0], h_state_action_encode.shape[-1]
        agent_ids = ops.eye(self.n_agents, dtype=ms.float32)
        agent_masks = (1 - agent_ids)
        repeat_agent_ids = agent_ids.unsqueeze(0).repeat(bs, 1, 1)
        repeated_agent_masks = agent_masks.unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1, dim_h)
        repeated_h_state_action_encode = h_state_action_encode.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        h_state_action_encode = repeated_h_state_action_encode * repeated_agent_masks
        h_state_action_encode = h_state_action_encode.sum(axis=2)  # Sum across other agents

        repeated_states = states.unsqueeze(1).repeat(1, self.n_agents, 1)
        input_q = ops.cat([repeated_states, h_state_action_encode, repeat_agent_ids], axis=-1)
        input_v = states
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt
