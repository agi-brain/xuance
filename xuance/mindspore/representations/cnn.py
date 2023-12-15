import mindspore as ms
import mindspore.nn as nn
from typing import Sequence, Optional, Union, Callable
import numpy as np
from xuance.mindspore.utils.layers import ModuleType, mlp_block, cnn_block, gru_block


# directly returns the original observation
class Basic_Identical(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int]):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}

    def construct(self, observations: ms.tensor):
        return {'state': observations}


# process the input observations with stacks of MLP layers
class Basic_MLP(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize)
            layers.extend(mlp)
        return nn.SequentialCell(*layers)

    def construct(self, observations):
        return {'state': self.model(observations)}


# process the input observations with stacks of CNN layers
class Basic_CNN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(Basic_CNN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (filters[-1],)}
        self._transpose = ms.ops.Transpose()
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.SequentialCell(*layers)

    def construct(self, observations: ms.tensor):
        tensor_observation = self._transpose(observations, (0, 3, 1, 2)).astype("float32")
        return {'state': self.model(tensor_observation)}


class CoG_CNN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CoG_CNN, self).__init__()
        self.input_shape = (input_shape['image'][2], input_shape['image'][0], input_shape['image'][1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.SequentialCell(*layers)

    def construct(self, observations: np.ndarray):
        tensor_observation = ms.Tensor(np.transpose(observations['image'], (0, 3, 1, 2)), dtype=ms.float32)
        return {'state': self.model(tensor_observation)}


# process the input observations with stacks of MLP layers
class CoG_MLP(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CoG_MLP, self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (128 + 64 + 64,)}
        self.laser_model, self.pose_model, self.angle_model = self._create_network()

    def _create_network(self):
        laser_layers = []
        pose_layers = []
        angle_layers = []

        laser_mlp1, _ = mlp_block(self.input_shape['laser'][0], 128, self.normalize, self.activation, self.initialize)
        laser_mlp2, _ = mlp_block(128, 128, self.normalize, self.activation, self.initialize)

        pose_mlp1, _ = mlp_block(2, 64, self.normalize, self.activation, self.initialize)
        pose_mlp2, _ = mlp_block(64, 64, self.normalize, self.activation, self.initialize)

        angle_mlp1, _ = mlp_block(4, 64, self.normalize, self.activation, self.initialize)
        angle_mlp2, _ = mlp_block(64, 64, self.normalize, self.activation, self.initialize)

        laser_layers = laser_mlp1 + laser_mlp2
        pose_layers = pose_mlp1 + pose_mlp2
        angle_layers = angle_mlp1 + angle_mlp2
        return nn.SequentialCell(*laser_layers), nn.SequentialCell(*pose_layers), nn.SequentialCell(*angle_layers)

    def construct(self, observations: np.ndarray):
        tensor_laser = ms.Tensor(observations['laser'], dtype=ms.float32)
        tensor_pose = ms.Tensor(observations['pose'], dtype=ms.float32)
        tensor_goal = ms.Tensor(observations['goal'], dtype=ms.float32)
        laser_feature = self.laser_model(tensor_laser)
        # pose_feature = self.pose_model(ms.ops.Concat((tensor_pose[:,0:2],tensor_goal[:,0:2]),dim=-1))
        pose_feature = self.pose_model(tensor_pose[:, 0:2] - tensor_goal[:, 0:2])
        angle_feature = self.angle_model(ms.ops.Concat((tensor_pose[:, 2:4], tensor_goal[:, 2:4]), dim=-1))
        return {'state': ms.ops.Concat((laser_feature, pose_feature, angle_feature), dim=-1)}


class CoG_RNN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CoG_RNN, self).__init__()
        self.input_shape = input_shape
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (256,)}
        self.laser_model, self.goal_model, self.pose_model, self.fusion_model = self._create_network()

    def _create_network(self):
        goal_layers = []
        goal_mlp1, _ = mlp_block(self.input_shape['goal'][0], 256, self.normalize, self.activation, self.initialize)
        goal_mlp2, _ = mlp_block(256, 256, self.normalize, nn.Tanh, self.initialize)
        goal_layers = goal_mlp1 + goal_mlp2

        laser_gru = gru_block(self.input_shape['laser'][1], 256, initialize=self.initialize)
        pose_gru = gru_block(self.input_shape['pose'][1], 256, initialize=self.initialize)
        fusion_mlp, _ = mlp_block(512, 256, None, self.activation, self.initialize)

        aux_mlp1 = mlp_block(512, 256, None, self.activation, self.initialize)
        aux_mlp2 = mlp_block(256, 2, None, self.activation, self.initialize)
        return laser_gru, nn.SequentialCell(*goal_layers), pose_gru, nn.SequentialCell(*fusion_mlp)

    def construct(self, observations: np.ndarray):
        tensor_laser = ms.Tensor(observations['laser'], dtype=ms.float32)
        tensor_pose = ms.Tensor(observations['pose'], dtype=ms.float32)
        tensor_goal = ms.Tensor(observations['goal'], dtype=ms.float32)

        _, laser_feature = self.laser_model(tensor_laser)
        goal_feature = self.goal_model(tensor_goal)
        _, pose_feature = self.pose_model(tensor_pose)

        laser_feature = laser_feature[0]
        pose_feature = pose_feature[0]
        fusion_feature = self.fusion_model(ms.ops.Concat((laser_feature, pose_feature), dim=-1))

        return {'state': fusion_feature * goal_feature}


class C_DQN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(C_DQN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.output_shapes = {'state': (filters[-1],)}
        self._transpose = ms.ops.Transpose()
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.SequentialCell(*layers)

    def construct(self, observations: ms.tensor):
        tensor_observation = self._transpose(observations, (0, 3, 1, 2)).astype("float32")
        return {'state': self.model(tensor_observation)}


class L_DQN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 output_shape: int,
                 dropout: float = 0,
                 initialize: Optional[Callable[..., ms.Tensor]] = None):
        super(L_DQN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout = dropout
        self.initialize = initialize
        self.model = self._create_network()

    def _create_network(self):
        input_shape = self.input_shape
        lstm = lstm_block(input_shape, self.output_shape, self.dropout, self.initialize)
        return lstm

    def construct(self, observations: ms.tensor):
        tensor_observation = self._transpose(observations, (0, 3, 1, 2)).astype("float32")
        return {'state': self.model(tensor_observation)}


class CL_DQN(nn.Cell):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 output_shape: int,
                 dropout: float = 0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None):
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
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            cnn, input_shape = cnn_block(input_shape, f, k, s, self.normalize, self.activation, self.initialize)
            layers.extend(cnn)
        layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(
            lstm_block(input_shape, self.output_shape, self.dropout, self.initialize)
            )
        return nn.SequentialCell(*layers)

    def construct(self, observations: ms.tensor):
        tensor_observation = self._transpose(observations, (0, 3, 1, 2)).astype("float32")
        return {'state': self.model(tensor_observation)}