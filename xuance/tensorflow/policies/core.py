import numpy as np
from tensorflow.keras.activations import softmax
from xuance.common import Sequence, Optional, Union, Callable
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
    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the output of the actor.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
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
        self.dist.set_param(logits=logits)
        return logits


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
        probs = softmax(logits)
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

    def call(self, x: Union[Tensor, np.ndarray], **kwargs):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        mu_ = self.mu(x)
        self.dist.set_param(mu_, tf.exp(self.logstd))
        return mu_


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
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_base, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.dim_q_input = (dim_utility_hidden + self.dim_action) * self.n_agents
        self.dim_v_input = dim_utility_hidden * self.n_agents

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
    def call(self, hidden_states_n, actions_n=None, **kwargs):
        input_q = tf.reshape(tf.concat([hidden_states_n, actions_n], axis=-1), [-1, self.dim_q_input])
        input_v = tf.reshape(hidden_states_n, [-1, self.dim_v_input])
        q_jt = self.Q_jt(input_q)
        v_jt = self.V_jt(input_v)
        return q_jt, v_jt


class QTRAN_alt(QTRAN_base):
    def __init__(self, dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden):
        super(QTRAN_alt, self).__init__(dim_state, dim_action, dim_hidden, n_agents, dim_utility_hidden)

    def counterfactual_values(self, q_self_values, q_selected_values):
        q_repeat = tf.tile(tf.expand_dims(q_selected_values, axis=1), multiples=(1, self.n_agents, 1, self.dim_action))
        counterfactual_values_n = q_repeat.numpy()
        for agent in range(self.n_agents):
            counterfactual_values_n[:, agent, agent] = q_self_values[:, agent, :].numpy()
        counterfactual_values_n = tf.convert_to_tensor(counterfactual_values_n)
        return tf.reduce_sum(counterfactual_values_n, axis=2)

    def counterfactual_values_hat(self, hidden_states_n, actions_n):
        action_repeat = tf.tile(tf.expand_dims(actions_n, axis=2), multiples=(1, 1, self.dim_action, 1))
        action_self_all = tf.expand_dims(tf.eye(self.dim_action), axis=0).numpy()
        action_counterfactual_n = tf.tile(tf.expand_dims(action_repeat, axis=2), multiples=(
            1, 1, self.n_agents, 1, 1)).numpy()  # batch * N * N * dim_a * dim_a
        q_n = []
        for agent in range(self.n_agents):
            action_counterfactual_n[:, agent, agent, :, :] = action_self_all
            q_actions = []
            for a in range(self.dim_action):
                input_a = tf.convert_to_tensor(action_counterfactual_n[:, :, agent, a, :])
                q, _ = self.call(hidden_states_n, input_a)
                q_actions.append(q)
            q_n.append(tf.expand_dims(tf.concat(q_actions, axis=-1), axis=1))
        return tf.concat(q_n, axis=1)
