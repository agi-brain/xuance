from xuanpolicy.torch.representations import *


class Basic_RNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs
                 ):
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
        self.mlp, self.rnn = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.fc_hidden_sizes:
            mlp_layer, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                               self.device)
            layers.extend(mlp_layer)
        if self.lstm:
            rnn_layer = lstm_block(self.fc_hidden_sizes[-1], self.recurrent_hidden_size, self.N_recurrent_layer,
                                   self.dropout, self.initialize, self.device)
        else:
            rnn_layer = gru_block(self.fc_hidden_sizes[-1], self.recurrent_hidden_size, self.N_recurrent_layer,
                                  self.dropout, self.initialize, self.device)
        return nn.Sequential(*layers), rnn_layer

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor = None):
        mlp_output = self.mlp(x)
        self.rnn.flatten_parameters()
        if self.lstm:
            output, (hn, cn) = self.rnn(mlp_output, (h, c))
            return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": cn.detach()}
        else:
            output, hn = self.rnn(mlp_output, h)
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


class CoG_RNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CoG_RNN, self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (256,)}
        self.laser_model, self.goal_model, self.pose_model, self.fusion_model = self._create_network()

    def _create_network(self):
        goal_layers = []
        goal_mlp1, _ = mlp_block(self.input_shape['goal'][0], 256, self.normalize, self.activation, self.initialize,
                                 self.device)
        goal_mlp2, _ = mlp_block(256, 256, self.normalize, nn.Tanh, self.initialize, self.device)
        goal_layers = goal_mlp1 + goal_mlp2

        laser_gru = gru_block(self.input_shape['laser'][1], 256, initialize=self.initialize, device=self.device)
        pose_gru = gru_block(self.input_shape['pose'][1], 256, initialize=self.initialize, device=self.device)
        fusion_mlp, _ = mlp_block(512, 256, None, self.activation, self.initialize, self.device)

        aux_mlp1 = mlp_block(512, 256, None, self.activation, self.initialize, self.device)
        aux_mlp2 = mlp_block(256, 2, None, self.activation, self.initialize, self.device)
        return laser_gru, nn.Sequential(*goal_layers), pose_gru, nn.Sequential(*fusion_mlp)

    def forward(self, observations: np.ndarray):
        tensor_laser = torch.as_tensor(observations['laser'], dtype=torch.float32, device=self.device)
        tensor_pose = torch.as_tensor(observations['pose'], dtype=torch.float32, device=self.device)
        tensor_goal = torch.as_tensor(observations['goal'], dtype=torch.float32, device=self.device)

        _, laser_feature = self.laser_model(tensor_laser)
        goal_feature = self.goal_model(tensor_goal)
        _, pose_feature = self.pose_model(tensor_pose)

        laser_feature = laser_feature[0]
        pose_feature = pose_feature[0]
        fusion_feature = self.fusion_model(torch.cat((laser_feature, pose_feature), dim=-1))

        return {'state': fusion_feature * goal_feature}
