from xuance.common import Sequence, Optional, Union, Callable, Tuple
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, mlp_block, gru_block, lstm_block, ModuleType


class Basic_RNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalize: Optional[Module] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
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

    def _create_network(self) -> Tuple[Module, Module, int]:
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

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None):
        if self.use_normalize:
            tensor_x = self.input_norm(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        else:
            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        mlp_output = self.mlp(tensor_x)
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

    def init_hidden_item(self, indexes: list, *rnn_hidden):
        zeros_size = (self.N_recurrent_layer, len(indexes), self.recurrent_hidden_size)
        if self.lstm:
            rnn_hidden[0][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            rnn_hidden[1][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            return rnn_hidden
        else:
            rnn_hidden[0][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            return rnn_hidden

    def get_hidden_item(self, i, *rnn_hidden):
        return (rnn_hidden[0][:, i], rnn_hidden[1][:, i]) if self.lstm else (rnn_hidden[0][:, i], None)
