from xuanpolicy.xuanpolicy_torch.representations import *


# directly returns the original observation
class Basic_Identical(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device
        self.model = nn.Sequential()

    def forward(self, observations: np.ndarray):
        state = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': state}


# process the input observations with stacks of MLP layers
class Basic_MLP(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         self.device)
            layers.extend(mlp)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}


# process the input observations with stacks of CNN layers
class Basic_CNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Basic_CNN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize,
                                         self.device)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=torch.float32,
                                             device=self.device)
        return {'state': self.model(tensor_observation)}


class CoG_CNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CoG_CNN, self).__init__()
        self.input_shape = (input_shape['image'][2], input_shape['image'][0], input_shape['image'][1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize,
                                         self.device)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(np.transpose(observations['image'], (0, 3, 1, 2)), dtype=torch.float32,
                                             device=self.device)
        return {'state': self.model(tensor_observation)}


# process the input observations with stacks of MLP layers
class CoG_MLP(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CoG_MLP, self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (128 + 64 + 64,)}
        self.laser_model, self.pose_model, self.angle_model = self._create_network()

    def _create_network(self):
        laser_layers = []
        pose_layers = []
        angle_layers = []

        laser_mlp1, _ = mlp_block(self.input_shape['laser'][0], 128, self.normalize, self.activation, self.initialize,
                                  self.device)
        laser_mlp2, _ = mlp_block(128, 128, self.normalize, self.activation, self.initialize, self.device)

        pose_mlp1, _ = mlp_block(2, 64, self.normalize, self.activation, self.initialize, self.device)
        pose_mlp2, _ = mlp_block(64, 64, self.normalize, self.activation, self.initialize, self.device)

        angle_mlp1, _ = mlp_block(4, 64, self.normalize, self.activation, self.initialize, self.device)
        angle_mlp2, _ = mlp_block(64, 64, self.normalize, self.activation, self.initialize, self.device)

        laser_layers = laser_mlp1 + laser_mlp2
        pose_layers = pose_mlp1 + pose_mlp2
        angle_layers = angle_mlp1 + angle_mlp2
        return nn.Sequential(*laser_layers), nn.Sequential(*pose_layers), nn.Sequential(*angle_layers)

    def forward(self, observations: np.ndarray):
        tensor_laser = torch.as_tensor(observations['laser'], dtype=torch.float32, device=self.device)
        tensor_pose = torch.as_tensor(observations['pose'], dtype=torch.float32, device=self.device)
        tensor_goal = torch.as_tensor(observations['goal'], dtype=torch.float32, device=self.device)
        laser_feature = self.laser_model(tensor_laser)
        # pose_feature = self.pose_model(torch.cat((tensor_pose[:,0:2],tensor_goal[:,0:2]),dim=-1))
        pose_feature = self.pose_model(tensor_pose[:, 0:2] - tensor_goal[:, 0:2])
        angle_feature = self.angle_model(torch.cat((tensor_pose[:, 2:4], tensor_goal[:, 2:4]), dim=-1))
        return {'state': torch.concat((laser_feature, pose_feature, angle_feature), dim=-1)}


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


class C_DQN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(C_DQN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize,
                                         self.device)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        tensor_observation = torch.as_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=torch.float32,
                                             device=self.device)
        return {'state': self.model(tensor_observation)}


class L_DQN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 output_shape: int,
                 dropout: float = 0,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(L_DQN, self).__init__()
        assert len(input_shape) == 1
        self.output_shape = output_shape
        self.device = device
        self.dropout = dropout
        self.initialize = initialize
        self.device = device
        self.model = self._create_network()

    def _create_network(self):
        input_shape = self.input_shape
        lstm = lstm_block(input_shape, self.output_shape, self.dropout, self.initialize, self.device)
        return lstm

    def forward(self, observations: np.ndarray):
        state = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': state}


class CL_DQN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 output_shape: int,
                 dropout: float = 0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CL_DQN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.output_shape = output_shape
        self.dropout = dropout
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize,
                                         self.device)
        lstm = lstm_block(input_shape, self.output_shape, self.dropout, self.initialize, self.device)
        return lstm

    def forward(self, observations: np.ndarray):
        state = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': state}