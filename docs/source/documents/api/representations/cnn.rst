CNN-based
=====================================

卷积神经网络主要用于处理图像输入数据，提取出特征向量，一般输入类型为多通道图像矩阵，输出多维向量，
名称为 cnn_block,其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py中。
实例化该类需要指定输入尺寸（input_shape），滤波方法（filter），核大小（kernel_size），步长（stride），
归一化方法（normalize），激活函数（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。


.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.representations.cnn.Basic_CNN(input_shape, kernels, strides, filters, normalize=None, initialize=None, activation=None, device=None)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence of int
    :param kernels: Size of the convolving kernel
    :type kernels: Sequence of int
    :param strides: Stride of the convolution.
    :type strides: a single number or a tuple of two ints
    :param filters: Number of channels produced by the convolution
    :type filters: Sequence of int
    :param normalize: The normalizer for the hidden variables of the representation.
    :type normalize: nn.Module
    :param initialize: The initializer of the parameters of the representation.
    :param activation: The activation function of each hidden layer.
    :type activation: nn.Module
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.representations.cnn.Basic_CNN._create_network()

    Create the convolutional neural netowrks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuanpolicy.torch.representations.cnn.Basic_CNN.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: numpy.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. py:class:: 
    xuanpolicy.torch.representations.cnn.AC_CNN_Atari(input_shape, kernels, strides, filters, normalize=None, initialize=None, activation=None, device=None)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence of int
    :param kernels: Size of the convolving kernel
    :type kernels: Sequence of int
    :param strides: Stride of the convolution.
    :type strides: a single number or a tuple of two ints
    :param filters: Number of channels produced by the convolution
    :type filters: Sequence of int
    :param normalize: The normalizer for the hidden variables of the representation.
    :type normalize: nn.Module
    :param initialize: The initializer of the parameters of the representation.
    :param activation: The activation function of each hidden layer.
    :type activation: nn.Module
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device
    :param fc_hidden_sizes: The sizes of the final fully connected hidden layers.
    :type device: Sequence of int

.. py:function:: 
    xuanpolicy.torch.representations.cnn.AC_CNN_Atari._init_layer(layer, gain=numpy.sqrt(2), bias=0.0)

    Initialize the weights and biases of the model.

    :param layer: A singe layer of the networks.
    :type layer: nn.Module
    :param gain: The gain of the weights with orthogonal initilizer, defualt is sqrt of 2.
    :type gain: float
    :param bias: The initial bias of the layer, defualt is sqrt of 0.
    :type bias: float
    :return: The initilized layer.
    :rtype: nn.Module

.. py:function:: 
    xuanpolicy.torch.representations.cnn.AC_CNN_Atari._create_network()

    Create the convolutional neural netowrks for actor-critic based algorithms and Atari tasks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuanpolicy.torch.representations.cnn.AC_CNN_Atari.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: numpy.ndarray
    :return: The features output by the representation model.
    :rtype: dict

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
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
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
                observations = observations / 255.0
                tensor_observation = torch.as_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=torch.float32,
                                                    device=self.device)
                return {'state': self.model(tensor_observation)}


        class AC_CNN_Atari(nn.Module):
            def __init__(self,
                        input_shape: Sequence[int],
                        kernels: Sequence[int],
                        strides: Sequence[int],
                        filters: Sequence[int],
                        normalize: Optional[ModuleType] = None,
                        initialize: Optional[Callable[..., torch.Tensor]] = None,
                        activation: Optional[ModuleType] = None,
                        device: Optional[Union[str, int, torch.device]] = None,
                        fc_hidden_sizes: Sequence[int] = ()):
                super(AC_CNN_Atari, self).__init__()
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
                self.kernels = kernels
                self.strides = strides
                self.filters = filters
                self.normalize = normalize
                self.initialize = initialize
                self.activation = activation
                self.device = device
                self.fc_hidden_sizes = fc_hidden_sizes
                self.output_shapes = {'state': (fc_hidden_sizes[-1],)}
                self.model = self._create_network()

            def _init_layer(self, layer, gain=np.sqrt(2), bias=0.0):
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, bias)
                return layer

            def _create_network(self):
                layers = []
                input_shape = self.input_shape
                for k, s, f in zip(self.kernels, self.strides, self.filters):
                    cnn, input_shape = cnn_block(input_shape, f, k, s, None, self.activation, None, self.device)
                    cnn[0] = self._init_layer(cnn[0])
                    layers.extend(cnn)
                layers.append(nn.Flatten())
                input_shape = (np.prod(input_shape, dtype=np.int), )
                for h in self.fc_hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, self.activation, None, self.device)
                    mlp[0] = self._init_layer(mlp[0])
                    layers.extend(mlp)
                return nn.Sequential(*layers)

            def forward(self, observations: np.ndarray):
                observations = observations / 255.0
                tensor_observation = torch.as_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=torch.float32,
                                                    device=self.device)
                return {'state': self.model(tensor_observation)}


  .. group-tab:: TensorFlow

    .. code-block:: python3

  .. group-tab:: MindSpore

    .. code-block:: python3
