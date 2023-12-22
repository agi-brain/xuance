CNN-based
=====================================

Convolutional Neural Networks (CNNs) are mainly used for processing image input data to extract feature vectors.
They usually take multi-channel image matrices as input and output multi-dimensional vectors.
The CNN block is defined in `./xuance/torch/utils/layers.py`, `./xuance/tensorflow/utils/layers.py` and `./xuance/mindspore/utils/layers.py`.

To instantiate this class, you need to specify the input size (`input_shape`), the filtering method (`filter`), the kernel size (`kernel_size`), the stride (`stride`), the normalization method (`normalize`), the activation function (`activation`), and the initialization method (`initialize`).

When implementing this class in PyTorch, you also need to specify the device type (`device`) to determine whether the model runs on CPU or GPU.


.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.representations.cnn.Basic_CNN(input_shape, kernels, strides, filters, normalize=None, initialize=None, activation=None, device)

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
    xuance.torch.representations.cnn.Basic_CNN._create_network()

    Create the convolutional neural netowrks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuance.torch.representations.cnn.Basic_CNN.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: np.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. py:class:: 
    xuance.torch.representations.cnn.AC_CNN_Atari(input_shape, kernels, strides, filters, normalize=None, initialize=None, activation=None, device=None)

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
    :type fc_hidden_sizes: Sequence of int

.. py:function:: 
    xuance.torch.representations.cnn.AC_CNN_Atari._init_layer(layer, gain=numpy.sqrt(2), bias=0.0)

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
    xuance.torch.representations.cnn.AC_CNN_Atari._create_network()

    Create the convolutional neural netowrks for actor-critic based algorithms and Atari tasks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuance.torch.representations.cnn.AC_CNN_Atari.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: np.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
    xuance.tensorflow.representations.cnn.Basic_CNN(input_shape, kernels, strides, filters, normalize=None, initialize=None, activation=None, device=None)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence of int
    :param kernels: Size of the convolving kernel
    :type kernels: Sequence of int
    :param strides: Stride of the convolution.
    :type strides: a single number or a tuple of two ints
    :param filters: Number of channels produced by the convolution
    :type filters: Sequence of int
    :param normalize: The normalizer for the hidden variables of the representation.
    :type normalize: tk.Model
    :param initialize: The initializer of the parameters of the representation.
    :param activation: The activation function of each hidden layer.
    :type activation: tk.Model
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function::
    xuance.tensorflow.representations.cnn.Basic_CNN._create_network()

    Create the convolutional neural netowrks.

    :return: The neural network module.
    :rtype: tk.Model

.. py:function::
    xuance.tensorflow.representations.cnn.Basic_CNN.call(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: np.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.representations.cnn.Basic_CNN(input_shape, kernels, strides, filters, normalize, initialize, activation)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param kernels: Size of the convolving kernel.
  :type kernels: Sequence of int
  :param strides: Stride of the convolution.
  :type strides: a single number or a tuple of two ints
  :param filters: Number of channels produced by the convolution.
  :type filters: Sequence of int
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.representations.cnn.Basic_CNN._create_network()

  Create the convolutional neural netowrks.

  :return: The neural network module.
  :rtype: nn.Cell

.. py:function::
  xuance.mindspore.representations.cnn.Basic_CNN.construct(observations)

  Calculate feature representation of the input observations.

  :param observations: The original observation variables.
  :type observations: ms.Tensor
  :return: The features output by the representation model.
  :rtype: dict

.. py:class::
  xuance.mindspore.representations.cnn.AC_CNN_Atari(input_shape, kernels, strides, filters, normalize, initialize, activation, fc_hidden_sizes)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param kernels: Size of the convolving kernel.
  :type kernels: Sequence of int
  :param strides: Stride of the convolution.
  :type strides: a single number or a tuple of two ints
  :param filters: Number of channels produced by the convolution.
  :type filters: Sequence of int
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell
  :param fc_hidden_sizes: The sizes of the final fully connected hidden layers.
  :type fc_hidden_sizes: list

.. py:function::
  xuance.mindspore.representations.cnn.AC_CNN_Atari._init_layer(layer, gain, bias)

  Initialize the weights and biases of the model.

  :param layer: A singe layer of the networks.
  :type layer: nn.Cell
  :param gain: The gain of the weights with orthogonal initilizer, defualt is sqrt of 2.
  :type gain: float
  :param bias: The initial bias of the layer, defualt is sqrt of 0.
  :type bias: float
  :return: The initilized layer.
  :rtype: nn.Cell

.. py:function::
  xuance.mindspore.representations.cnn.AC_CNN_Atari._create_network()

  Create the convolutional neural netowrks for actor-critic based algorithms and Atari tasks.

  :return: The neural network module.
  :rtype: nn.Cell

.. py:function::
  xuance.mindspore.representations.cnn.AC_CNN_Atari.construct(observations)

  Calculate feature representation of the input observations.

  :param observations: The original observation variables.
  :type observations: ms.Tensor
  :return: The features output by the representation model.
  :rtype: dict

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
  .. group-tab:: PyTorch
    
    .. code-block:: python

        from xuance.torch.representations import *

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

    .. code-block:: python

        from xuance.tensorflow.representations import *


        class Basic_CNN(tk.Model):
            def __init__(self,
                         input_shape: Sequence[int],
                         kernels: Sequence[int],
                         strides: Sequence[int],
                         filters: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initialize: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu"):
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
                layers.append(tfa.layers.AdaptiveMaxPooling2D((1, 1)))
                layers.append(tk.layers.Flatten())
                return tk.Sequential(*layers)

            def call(self, observations: np.ndarray, **kwargs):
                with tf.device(self.device):
                    tensor_observation = tf.convert_to_tensor(np.transpose(observations, (0, 3, 1, 2)), dtype=tf.float32)
                    return {'state': self.model(tensor_observation)}



  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.representations import *
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


        class AC_CNN_Atari(nn.Cell):
            def __init__(self,
                         input_shape: Sequence[int],
                         kernels: Sequence[int],
                         strides: Sequence[int],
                         filters: Sequence[int],
                         normalize: Optional[ModuleType] = None,
                         initialize: Optional[Callable[..., ms.Tensor]] = None,
                         activation: Optional[ModuleType] = None,
                         fc_hidden_sizes: Sequence[int] = ()):
                super(AC_CNN_Atari, self).__init__()
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
                self.kernels = kernels
                self.strides = strides
                self.filters = filters
                self.normalize = normalize
                self.initialize = initialize
                self.activation = activation
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
                    cnn, input_shape = cnn_block(input_shape, f, k, s, None, self.activation, None)
                    cnn[0] = self._init_layer(cnn[0])
                    layers.extend(cnn)
                layers.append(nn.Flatten())
                input_shape = (np.prod(input_shape, dtype=np.int), )
                for h in self.fc_hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h, None, self.activation, None)
                    mlp[0] = self._init_layer(mlp[0])
                    layers.extend(mlp)
                return nn.SequentialCell(*layers)

            def construct(self, observations: np.ndarray):
                observations = observations / 255.0
                tensor_observation = ms.tensor(np.transpose(observations, (0, 3, 1, 2))).astype(ms.float32)
                return {'state': self.model(tensor_observation)}

