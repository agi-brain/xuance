RNN-based
=====================================

循环神经网络主要用于处理时序信号信息，提取出当前时序信号的特征向量。
根据使用场景差异，本软件提供两种循环神经网路模块：gru_block和lstm_block，
其定义均位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），
剪枝方法（droupout），初始化方法（initialize）。同样地，在pytorch下实现还需指定设备类型（device），
以确定模型在CPU上运行还是GPU。

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN(input_shape, hidden_sizes, normalize=None, initialize=None, activation=None, device=None, kwargs)

    The ``hidden_sizes`` is a dict input, which contains "fc_hidden_sizes" and "fc_hidden_sizes".
    The "fc_hidden_sizes" is the sizes of the fully connected layers before rnn layers.
    The "recurrent_hidden_size" is the size of recurrent layer.

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence[int]
    :param hidden_sizes: The sizes of the hidden layers.
    :type hidden_sizes: dict
    :param device: Choose CPU or GPU to train the model.
    :param normalize: The normalizer for the hidden variables of the representation.
    :type normalize: nn.Module
    :param initialize: The initializer of the parameters of the representation.
    :param activation: The activation function of each hidden layer.
    :type activation: nn.Module
    :type device: str, int, torch.device
    :type N_recurrent_layers: 
    :type N_recurrent_layers:
    :type dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
    :type dropout: float
    :type rnn: Choose the rnn cell: "GRU" or "LSTM.
    :type rnn: str

.. py:function:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN._create_network()

    Create the recurrent neural netowrks.

    :return: The RNN neural network with fully connected layers, the RNN layer, and the shape of the RNN hidden states.
    :rtype: nn.Module, nn.Module, int

.. py:function:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN.forward(x, h, c=None)

    Calculate feature representation of the inputs.

    :param x: The input data.
    :type x: torch.Tensor
    :param h: The hidden states of the recurrent layers at last step.
    :type h: torch.Tensor
    :param c: The cell states of the LSTM layers at last step.
    :type c: torch.Tensor
    :return: The features output by the representation model, new hidden states, and new cell states.
    :rtype: dict

.. py:function:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN.init_hidden(batch)

    Initialize a batch of RNN hidden states.

    :param batch: The size of the batch.
    :type batch: int
    :return: The initialized hidden states.
    :rtype: torch.Tensor

.. py:function:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN.init_hidden_item(i, rnn_hidden)

    Initialize a slice of hidden states from the given RNN hidden states.

    :param i: The index of the slice.
    :type i: int
    :param rnn_hidden: The RNN hidden states.
    :type i: torch.Tensor
    :return: The initialized hidden states.
    :rtype: torch.Tensor

.. py:function:: 
    xuanpolicy.torch.representations.rnn.Basic_RNN.get_hidden_item(i, rnn_hidden)

    Get a slice of hidden states from the given RNN hidden states.

    :param i: The index of the slice.
    :type i: int
    :param rnn_hidden: The RNN hidden states.
    :type i: torch.Tensor
    :return: The selected hidden states.
    :rtype: torch.Tensor

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

源码
-----------------

.. tabs::
  
  .. group-tab:: PyTorch
    
    .. code-block:: python3

        from xuanpolicy.torch.representations import *

        class Basic_RNN(nn.Module):
            def __init__(self,
                        input_shape: Sequence[int],
                        hidden_sizes: dict,
                        normalize: Optional[nn.Module] = None,
                        initialize: Optional[Callable[..., torch.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        device: Optional[Union[str, int, torch.device]] = None,
                        **kwargs):
                super(Basic_RNN, self).__init__()
                self.input_shape = input_shape
                self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
                self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
                self.N_recurrent_layer = kwargs["N_recurrent_layers"]
                self.dropout = kwargs["dropout"]
                self.lstm = True if kwargs["rnn"] == "LSTM" else False
                self.normalize = normalize
                self.initialize = initialize
                self.activation = activation
                self.device = device
                self.output_shapes = {'state': (hidden_sizes["recurrent_hidden_size"],)}
                self.mlp, self.rnn, output_dim = self._create_network()
                if self.normalize is not None:
                    self.use_normalize = True
                    self.input_norm = self.normalize(input_shape, device=device)
                    self.norm_rnn = self.normalize(output_dim, device=device)
                else:
                    self.use_normalize = False

            def _create_network(self) -> Tuple[nn.Module, nn.Module, int]:
                layers = []
                input_shape = self.input_shape
                for h in self.fc_hidden_sizes:
                    mlp_layer, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                                    device=self.device)
                    layers.extend(mlp_layer)
                if self.lstm:
                    rnn_layer, input_shape = lstm_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                                        self.dropout, self.initialize, self.device)
                else:
                    rnn_layer, input_shape = gru_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                                    self.dropout, self.initialize, self.device)
                return nn.Sequential(*layers), rnn_layer, input_shape

            def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor = None):
                mlp_output = self.mlp(self.input_norm(x)) if self.use_normalize else self.mlp(x)
                self.rnn.flatten_parameters()
                if self.lstm:
                    output, (hn, cn) = self.rnn(mlp_output, (h, c))
                    if self.use_normalize:
                        output = self.norm_rnn(output)
                    return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": cn.detach()}
                else:
                    output, hn = self.rnn(mlp_output, h)
                    if self.use_normalize:
                        output = self.norm_rnn(output)
                    return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": None}

            def init_hidden(self, batch):
                hidden_states = torch.zeros(size=(self.N_recurrent_layer, batch, self.recurrent_hidden_size)).to(self.device)
                cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
                return hidden_states, cell_states

            def init_hidden_item(self, i, *rnn_hidden):
                if self.lstm:
                    rnn_hidden[0][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
                    rnn_hidden[1][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
                    return rnn_hidden
                else:
                    rnn_hidden[0][:, i] = torch.zeros(size=(self.N_recurrent_layer, self.recurrent_hidden_size)).to(self.device)
                    return rnn_hidden

            def get_hidden_item(self, i, *rnn_hidden):
                return (rnn_hidden[0][:, i], rnn_hidden[1][:, i]) if self.lstm else (rnn_hidden[0][:, i], None)


  .. group-tab:: TensorFlow

    .. code-block:: python3

  .. group-tab:: MindSpore

    .. code-block:: python3
