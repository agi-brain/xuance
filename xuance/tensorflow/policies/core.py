import numpy as np
from gymnasium.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union, Dict
from xuance.tensorflow import tf, tk, Module, Tensor
from xuance.tensorflow.utils import mlp_block, gru_block, lstm_block, ModuleType
from xuance.tensorflow.utils import CategoricalDistribution, DiagGaussianDistribution, ActivatedDiagGaussianDistribution


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
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
        super(BasicQhead, self).__init__()
        layers = []
        self.state_dim = state_dim
        self.n_actions = n_actions
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions, None, None, initialize)[0])
        self.model = tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the output of the Q network.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
        """
        input_shape = x.shape
        x_flat = tf.reshape(x, (-1, input_shape[-1]))
        y_flat = self.model(x_flat)
        return tf.reshape(y_flat, input_shape[:-1] + (self.n_actions, ))


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
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
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
        self.a_model = tk.Sequential(a_layers)
        self.v_model = tk.Sequential(v_layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the dueling Q-values.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.

        Returns:
            q: The dueling Q-values.
        """
        v = self.v_model(x)
        a = self.a_model(x)
        q = v + (a - tf.expand_dims(tf.reduce_mean(a, axis=-1), axis=-1))
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
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
        super(C51Qhead, self).__init__()
        self.action_dim = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, initialize)[0])
        self.model = tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the discrete action distributions.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
        Returns:
            dist_probs: The probability distribution of the discrete actions.
        """
        dist_logits = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
        dist_probs = tf.nn.softmax(dist_logits, axis=-1)
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
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None):
        super(QRDQNhead, self).__init__()
        self.action_dim = n_actions
        self.atom_num = atom_num
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], n_actions * atom_num, None, None, None)[0])
        self.model = tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the quantiles of the distribution.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
        Returns:
            quantiles: The quantiles of the action distribution.
        """
        quantiles = tf.reshape(self.model(x), [-1, self.action_dim, self.atom_num])
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
        self.output_dim = kwargs["action_dim"]
        self.model = tk.Sequential(fc_layer)
        self.rnn_layer.build(input_shape=(None, None, kwargs["input_dim"]))

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """Returns the rnn hidden and Q-values via RNN networks."""
        if self.lstm:
            rnn_output, hn, cn = self.rnn_layer(x)
            fc_input_shape = rnn_output.shape
            fc_input = tf.reshape(x, [-1, fc_input_shape[-1]])
            fc_output = self.model(fc_input)
            return hn, cn, tf.reshape(fc_output, fc_input_shape[:-1] + (self.output_dim, ))
        else:
            rnn_output, hn = self.rnn_layer(x)
            fc_input_shape = rnn_output.shape
            fc_input = tf.reshape(x, [-1, fc_input_shape[-1]])
            fc_output = self.model(fc_input)
            return hn, tf.reshape(fc_output, fc_input_shape[:-1] + (self.output_dim,))


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
                 activation: Optional[tk.layers.Layer] = None,
                 activation_action: Optional[tk.layers.Layer] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, activation_action, initialize)[0])
        self.model = tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], avail_actions: Optional[Tensor] = None, **kwargs):
        """
        Returns the output of the actor.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
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
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, initialize)[0])
        self.model = tk.Sequential(layers)
        self.dist = CategoricalDistribution(action_dim)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], avail_actions: Optional[Tensor] = None, **kwargs):
        """
        Returns the stochastic distribution over all discrete actions.

        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
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

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], avail_actions: Optional[Tensor] = None, **kwargs):
        """
        Returns the stochastic distribution over all discrete actions.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
            avail_actions (Optional[Tensor]): The actions mask values when use actions mask, default is None.

        Returns:
            self.dist: CategoricalDistribution(action_dim), a distribution over all discrete actions.
        """
        logits = self.model(x)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        probs = tf.nn.softmax(logits, axis=-1)
        self.dist.set_param(probs=probs)
        return probs


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
        self.mu = tk.Sequential(layers)
        self.logstd = self.add_weight(name="log_of_std",
                                      shape=(action_dim,),
                                      initializer=tf.ones,
                                      trainable=True)
        # self.logstd = tf.Variable(tf.zeros((action_dim,)) - 1, trainable=True)
        self.dist = DiagGaussianDistribution(action_dim)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.

        Returns:
            mu_: The mean variable of the Gaussian distribution.
        """
        mu_ = self.mu(x)
        return mu_

    def distribution(self, mu: Tensor, std: Tensor):
        self.dist.set_param(mu=mu, std=std)
        return self.dist


class CriticNet(Module):
    """
    The actor network for categorical policy, which outputs a distribution over all discrete actions.

    Args:
        input_dim (int): The input state dimension.
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
                 activation: Optional[tk.layers.Layer] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (input_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = tk.Sequential(layers)

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the output of the Q network.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
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
        self.out = tk.Sequential(layers)
        self.out_mu = tk.layers.Dense(units=action_dim, activation=None, input_shape=(hidden_sizes[-1],))
        self.out_log_std = tk.layers.Dense(units=action_dim, activation=None, input_shape=(hidden_sizes[-1],))
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action)
        self.out_mu.build(input_shape=(None, hidden_sizes[-1]))
        self.out_log_std.build(input_shape=(None, hidden_sizes[-1]))

    @tf.function
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        output = self.out(x)
        mu = self.out_mu(output)
        log_std = tf.clip_by_value(self.out_log_std(output), -20, 2)
        std = tf.exp(log_std)
        self.dist.set_param(mu, std)
        return mu


class VDN_mixer(Module):
    """
    The value decomposition networks mixer. (Additivity)
    """

    def __init__(self):
        super(VDN_mixer, self).__init__()

    @tf.function
    def call(self, values_n, states=None, **kwargs):
        return tf.reduce_sum(values_n, axis=1)


class QMIX_mixer(Module):
    """
    The QMIX mixer. (Monotonicity)

    Args:
        dim_state (int): The dimension of global state.
        dim_hidden (int): The size of rach hidden layer.
        dim_hypernet_hidden (int): The size of rach hidden layer for hyper network.
        n_agents (int): The number of agents.
    """

    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents):
        super(QMIX_mixer, self).__init__()
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_hypernet_hidden = dim_hypernet_hidden
        self.n_agents = n_agents
        # self.hyper_w_1 = nn.Linear(self.dim_state, self.dim_hidden * self.n_agents)
        # self.hyper_w_2 = nn.Linear(self.dim_state, self.dim_hidden)
        self.hyper_w_1 = tk.Sequential([tk.layers.Dense(units=self.dim_hypernet_hidden,
                                                        activation=tk.layers.Activation('relu'),
                                                        input_shape=(self.dim_state,)),
                                        tk.layers.Dense(units=self.dim_hidden * self.n_agents,
                                                        input_shape=(self.dim_hypernet_hidden,))])
        self.hyper_w_2 = tk.Sequential([tk.layers.Dense(units=self.dim_hypernet_hidden,
                                                        activation=tk.layers.Activation('relu'),
                                                        input_shape=(self.dim_state,)),
                                        tk.layers.Dense(units=self.dim_hidden,
                                                        input_shape=(self.dim_hypernet_hidden,))])

        self.hyper_b_1 = tk.layers.Dense(units=self.dim_hidden, input_shape=(self.dim_state,))
        self.hyper_b_2 = tk.Sequential([tk.layers.Dense(units=self.dim_hypernet_hidden,
                                                        activation=tk.layers.Activation('relu'),
                                                        input_shape=(self.dim_state,)),
                                        tk.layers.Dense(units=1, input_shape=(self.dim_hypernet_hidden,))])

    @tf.function
    def call(self, values_n, states=None, **kwargs):
        """
        Returns the total Q-values for multi-agent team.

        Parameters:
            values_n: The individual values for agents in team.
            states: The global states.

        Returns:
            q_tot: The total Q-values for the multi-agent team.
        """
        states = tf.reshape(states, [-1, self.dim_state])
        agent_qs = tf.reshape(values_n, [-1, 1, self.n_agents])
        # First layer
        w_1 = tf.abs(self.hyper_w_1(states))
        w_1 = tf.reshape(w_1, [-1, self.n_agents, self.dim_hidden])
        b_1 = self.hyper_b_1(states)
        b_1 = tf.reshape(b_1, [-1, 1, self.dim_hidden])
        hidden = tf.nn.elu(tf.linalg.matmul(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = tf.abs(self.hyper_w_2(states))
        w_2 = tf.reshape(w_2, [-1, self.dim_hidden, 1])
        b_2 = self.hyper_b_2(states)
        b_2 = tf.reshape(b_2, [-1, 1, 1])
        # Compute final output
        y = tf.linalg.matmul(hidden, w_2) + b_2
        # Reshape and return
        q_tot = tf.reshape(y, [-1, 1])
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
        self.ff_net = tk.Sequential([tk.layers.Dense(input_shape=(self.dim_input,), units=self.dim_hidden,
                                                     activation=tk.layers.Activation('relu')),
                                     tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                                     activation=tk.layers.Activation('relu')),
                                     tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                                     activation=tk.layers.Activation('relu')),
                                     tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)])
        self.ff_net_bias = tk.Sequential([tk.layers.Dense(input_shape=(self.dim_state,), units=self.dim_hidden,
                                                          activation=tk.layers.Activation('relu')),
                                          tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)])

    @tf.function
    def call(self, values_n, states=None, **kwargs):
        """
        Returns the feedforward total Q-values.

        Parameters:
            values_n: The individual Q-values.
            states: The global states.
        """
        states = tf.reshape(states, [-1, self.dim_state])
        agent_qs = tf.reshape(values_n, [-1, self.n_agents])
        inputs = tf.concat([agent_qs, states], axis=-1)
        out_put = self.ff_net(inputs)
        bias = self.ff_net_bias(states)
        y = out_put + bias
        q_tot = tf.reshape(y, [-1, 1])
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
        device: Optional[Union[str, int, torch.device]]: The calculating device.
    """
    def __init__(self,
                 dim_state: int = 0,
                 action_space: Dict[str, Discrete] = None,
                 dim_hidden: int = 32,
                 n_agents: int = 1,
                 dim_utility_hidden: int = 1,
                 use_parameter_sharing: bool = False,):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.action_space = action_space
        self.n_actions_list = [a_space.n for a_space in action_space.values()]
        self.n_actions_max = max(self.n_actions_list)
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing

        self.dim_q_input = self.dim_state + dim_utility_hidden + self.n_actions_max
        self.dim_v_input = self.dim_state

        linear_Q_jt = [tk.layers.Dense(input_shape=(self.dim_q_input,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.Q_jt = tk.Sequential(linear_Q_jt)
        linear_V_jt = [tk.layers.Dense(input_shape=(self.dim_v_input,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=self.dim_hidden,
                                       activation=tk.layers.Activation('relu')),
                       tk.layers.Dense(input_shape=(self.dim_hidden,), units=1)]
        self.V_jt = tk.Sequential(linear_V_jt)

    @tf.function
    def call(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor, **kwargs):
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
        h_state_action_input = tf.concat([hidden_state_inputs, actions_onehot], axis=-1)
        h_state_action_encode = tf.reshape(self.action_encoding(h_state_action_input),
                                           [-1, self.n_agents, self.dim_ae_input])
        h_state_action_encode = h_state_action_encode.sum(axis=1)  # Sum across agents
        input_q = tf.concat([states, h_state_action_encode], axis=-1)
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
    def __init__(self,
                 dim_state: int = 0,
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

        self.Q_jt = tf.keras.Sequential([
            tk.layers.Dense(self.dim_hidden, input_shape=(self.dim_q_input,)),
            tk.layers.ReLU(),
            tk.layers.Dense(self.dim_hidden),
            tk.layers.ReLU(),
            tk.layers.Dense(self.n_actions_max)
        ])

        self.V_jt = tf.keras.Sequential([
            tk.layers.Dense(self.dim_hidden, input_shape=(self.dim_v_input,)),
            tk.layers.ReLU(),
            tk.layers.Dense(self.dim_hidden),
            tk.layers.ReLU(),
            tk.layers.Dense(1)
        ])

        self.dim_ae_input = dim_utility_hidden + self.n_actions_max

        self.action_encoding = tf.keras.Sequential([
            tk.layers.Dense(self.dim_ae_input, input_shape=(self.dim_ae_input,)),
            tk.layers.ReLU(),
            tk.layers.Dense(self.dim_ae_input)
        ])

    @tf.function
    def call(self, states: Tensor, hidden_state_inputs: Tensor, actions_onehot: Tensor, **kwargs):
        """Calculating the joint Q and V values.

        Parameters:
            states (Tensor): The global states.
            hidden_state_inputs (Tensor): The joint hidden states inputs for QTRAN network.
            actions_onehot (Tensor): The joint onehot actions for QTRAN network.

        Returns:
            q_jt (Tensor): The evaluated joint Q values.
            v_jt (Tensor): The evaluated joint V values.
        """
        h_state_action_input = tf.concat([hidden_state_inputs, actions_onehot], axis=-1)

        h_state_action_encode = self.action_encoding(h_state_action_input)
        bs = tf.shape(h_state_action_encode)[0]
        dim_h = tf.shape(h_state_action_encode)[-1]

        agent_ids = tf.eye(self.n_agents, dtype=tf.float32)
        agent_masks = 1.0 - agent_ids
        repeat_agent_ids = tf.tile(agent_ids[tf.newaxis, :, :], [bs, 1, 1])  # [bs, n_agents, n_agents]
        repeated_agent_masks = tf.tile(agent_masks[tf.newaxis, :, :, tf.newaxis], [bs, 1, 1, dim_h])

        repeated_h_state_action_encode = tf.tile(h_state_action_encode[:, :, tf.newaxis, :], [1, 1, self.n_agents, 1])
        h_state_action_encode_masked = repeated_h_state_action_encode * repeated_agent_masks
        h_state_action_encode_sum = tf.reduce_sum(h_state_action_encode_masked, axis=2)  # sum over other agents

        repeated_states = tf.tile(states[:, tf.newaxis, :], [1, self.n_agents, 1])
        input_q = tf.concat([repeated_states, h_state_action_encode_sum, repeat_agent_ids], axis=-1)

        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(states)

        return q_jt, v_jt
