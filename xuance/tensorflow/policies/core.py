import numpy as np
from xuance.common import Sequence, Optional, Union, Callable
from xuance.tensorflow import tf, tk, Module, Tensor
from xuance.tensorflow.utils import mlp_block, gru_block, lstm_block, ModuleType


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

    def call(self, x: Union[Tensor, np.ndarray]):
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

    def call(self, x: Union[Tensor, np.ndarray]):
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

    def call(self, x: Union[Tensor, np.ndarray]):
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
       device (Optional[Union[str, int, torch.device]]): The calculating device.
    """
    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 atom_num: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[tk.layers.Layer] = None,
                 initialize: Optional[tk.initializers.Initializer] = None,
                 activation: Optional[tk.layers.Layer] = None,
                 device: str = "cpu:0"):
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

    def call(self, x: Union[Tensor, np.ndarray]):
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
        self.model = tk.Sequential(*fc_layer)

    def call(self, x: Union[Tensor, np.ndarray]):
        """Returns the rnn hidden and Q-values via RNN networks."""
        if self.lstm:
            output, hn, cn = self.rnn_layer(x)
            return hn, cn, self.model(output)
        else:
            output, hn = self.rnn_layer(x)
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

    def call(self, x: Union[Tensor, np.ndarray]):
        """
        Returns the output of the actor.
        Parameters:
            x (Union[Tensor, np.ndarray]): The input tensor.
        """
        return self.model(x)


class CriticNet(Module):
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
                 activation: Optional[tk.layers.Layer] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + action_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize)[0])
        self.model = tk.Sequential(layers)

    def call(self, x: Union[Tensor, np.ndarray]):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)