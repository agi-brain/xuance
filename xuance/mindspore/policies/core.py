import mindspore as ms
import mindspore.nn as nn
from xuance.common import Sequence, Optional, Callable, Union
from xuance.mindspore import Tensor, Module, ops
from xuance.mindspore.utils import ModuleType, mlp_block, gru_block, lstm_block
from xuance.mindspore.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution


class BasicQhead(Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(BasicQhead, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x)


class DuelQhead(Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
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
        a_layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])

        self.a_model = nn.SequentialCell(*a_layers)
        self.v_model = nn.SequentialCell(*v_layers)

        self._mean = ms.ops.ReduceMean(keep_dims=True)

    def construct(self, x: ms.tensor):
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - self._mean(a))
        return q


class C51Qhead(Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(C51Qhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)
        self._softmax = ms.ops.Softmax(axis=-1)

    def construct(self, x: ms.tensor):
        dist_logits = self.model(x).view(-1, self.action_dim, self.atom_num)
        dist_probs = self._softmax(dist_logits)
        return dist_probs


class QRDQNhead(Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(QRDQNhead, self).__init__()
        self.action_dim = action_dim
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim * atom_num, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.tensor):
        return self.model(x).view(-1, self.action_dim, self.atom_num)


class BasicRecurrent(Module):
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

    def construct(self, x: ms.tensor, h: ms.tensor, c: ms.tensor = None):
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
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
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

    def construct(self, x: ms.tensor):
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
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
        super(CategoricalActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)
        self.softmax = nn.Softmax()
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
        probs = self.softmax(logits)
        try:
            self.dist.set_param(probs=probs)
        except:
            pass
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
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
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

    def construct(self, x: ms.Tensor):
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
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.Tensor):
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
        mu = self.out_mu(output)
        log_std = ops.clip_by_value(self.out_log_std(output), -20, 2)
        std = ops.exp(log_std)
        self.dist.set_param(mu, std)
        return self.dist


class VDN_mixer(Module):
    def __init__(self):
        super(VDN_mixer, self).__init__()
        self._sum = ms.ops.ReduceSum(keep_dims=False)

    def construct(self, values_n, states=None):
        return self._sum(values_n, 1)


class QMIX_mixer(Module):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents):
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
    def __init__(self, dim_state, dim_hidden, n_agents):
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
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.view(-1, self.n_agents)
        inputs = self._concat([agent_qs, states])
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = y.view(-1, 1)
        return q_tot


class QTRAN_base(Module):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

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
        self._concat = ms.ops.Concat(axis=-1)

    def construct(self, hidden_states_n, actions_n):
        input_q = self._concat([hidden_states_n, actions_n]).view(-1, self.dim_q_input)
        input_v = hidden_states_n.view(-1, self.dim_v_input)
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(q_selected_values, axis=1),
                                       (-1, self.n_agents, -1, self.dim_action))
        counterfactual_values_n = q_repeat
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :]
        return counterfactual_values_n.sum(axis=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = ms.ops.broadcast_to(ms.ops.expand_dims(actions_n, axis=2), (-1, -1, self.dim_action, -1))
        action_self_all = ms.ops.expand_dims(ms.ops.eye(self.dim_action, self.dim_action, ms.float32), axis=0)
        action_counterfactual_n = ms.ops.broadcast_to(ms.ops.expand_dims(action_repeat, axis=2),
                                                      (-1, -1, self.n_agents, -1, -1))  # batch * N * N * dim_a * dim_a

        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = action_counterfactual_n[:, :, agent, a, :]
                q, _ = self.construct(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(ms.ops.expand_dims(self._concat(q_actions), axis=1))
        return ms.ops.concat(q_n, axis=1)
