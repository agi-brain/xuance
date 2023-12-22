MLP-based
=====================================

The Multi-Layer Perceptron (MLP) is one of the simplest deep neural network models used for processing vector inputs.
Users can instantiate the MLP module according to their own needs, which is defined in the `./xuance/torch/utils/layers.py`, `./xuance/tensorflow/utils/layers.py` and `./xuance/mindspore/utils/layers.py` files with the class name `mlp_block`.

To instantiate this class, you need to specify the input dimension (`input_dim`), output dimension (`output_dim`), normalization method (`normalize`), activation function choice (`activation`), and initialization method (`initialize`).

When implementing this class in PyTorch, you also need to specify the device type (`device`) to determine whether the model runs on CPU or GPU.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class:: 
    xuance.torch.representations.mlp.Basic_Identical(input_shape, device)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence[int]
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.representations.mlp.Basic_Identical.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: np.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. py:class:: 
    xuance.torch.representations.mlp.Basic_MLP(input_shape, device)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence[int]
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.representations.mlp.Basic_MLP._create_network()

    Create the multi-layer perceptron netowrks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuance.torch.representations.mlp.Basic_MLP.forward(observations)

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
  xuance.tensorflow.representations.mlp.Basic_Identical(input_shape, device)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.representations.mlp.Basic_Identical.call(observations)

  Calculate feature representation of the input observations.

  :param observations: The original observation variables.
  :type observations: tf.Tensor
  :return: The features output by the representation model.
  :rtype: dict

.. py:class::
  xuance.tensorflow.representations.mlp.Basic_MLP(input_shape, hidden_sizes, normalize, initialize, activation, device)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: tk.Model
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: tf.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: tk.Model
  :param device: The calculating device.
  :type device: str

.. py:function::
  xuance.tensorflow.representations.mlp.Basic_MLP._create_network()

  Create the multi-layer perceptron netowrks.

  :return: The neural network module.
  :rtype: tk.Model

.. py:function::
  xuance.tensorflow.representations.mlp.Basic_MLP.call(observations)

  Calculate feature representation of the input observations.

  :param observations: The original observation variables.
  :type observations: tf.Tensor
  :return: The features output by the representation model.
  :rtype: dict

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.representations.mlp.Basic_Identical(input_shape)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]

.. py:function::
  xuance.mindspore.representations.mlp.Basic_Identical.construct(observations)

  Calculate feature representation of the input observations.

  :param observations: The original observation variables.
  :type observations: ms.Tensor
  :return: The features output by the representation model.
  :rtype: dict

.. py:class::
  xuance.mindspore.representations.mlp.Basic_MLP(input_shape, hidden_sizes, normalize, initialize, activation)

  :param input_shape: The shape of the input data.
  :type input_shape: Sequence[int]
  :param hidden_sizes: The sizes of the hidden layers.
  :type hidden_sizes: Sequence[int]
  :param normalize: The method of normalization.
  :type normalize: nn.Cell
  :param initialize: The initialization for the parameters of the networks.
  :type initialize: ms.Tensor
  :param activation: The choose of activation functions for hidden layers.
  :type activation: nn.Cell

.. py:function::
  xuance.mindspore.representations.mlp.Basic_MLP._create_network()

  Create the multi-layer perceptron netowrks.

  :return: The neural network module.
  :rtype: nn.Cell

.. py:function::
  xuance.mindspore.representations.mlp.Basic_MLP.construct(observations)

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
                                                device=self.device)
                    layers.extend(mlp)
                return nn.Sequential(*layers)

            def forward(self, observations: np.ndarray):
                tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                return {'state': self.model(tensor_observation)}

  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.tensorflow.representations import *


        class Basic_Identical(tk.Model):
            def __init__(self,
                         input_shape: Sequence[int],
                         device: str = "cpu"):
                super(Basic_Identical, self).__init__()
                self.input_shapes = input_shape
                self.output_shapes = {'state': (np.prod(input_shape),)}
                self.device = device
                self.model = tk.Sequential([tk.layers.Flatten()])

            def call(self, observations: np.ndarray, **kwargs):
                with tf.device(self.device):
                    state = tf.convert_to_tensor(observations, dtype=tf.float32)
                    return {'state': state}


        class Basic_MLP(tk.Model):
            def __init__(self,
                         input_shapes: Sequence[int],
                         hidden_sizes: Sequence[int],
                         normalize: Optional[tk.layers.Layer] = None,
                         initializer: Optional[tk.initializers.Initializer] = None,
                         activation: Optional[tk.layers.Layer] = None,
                         device: str = "cpu"):
                super(Basic_MLP, self).__init__()
                self.input_shapes = input_shapes
                self.hidden_sizes = hidden_sizes
                self.normalize = normalize
                self.initializer = initializer
                self.activation = activation
                self.device = device
                self.output_shapes = {'state': (hidden_sizes[-1],)}
                self.model = self._create_network()

            def _create_network(self):
                layers = [tk.layers.Flatten()]
                input_shapes = (np.prod(self.input_shapes),)
                for h in self.hidden_sizes:
                    mlp, input_shapes = mlp_block(input_shapes[0], h, self.normalize, self.activation, self.initializer,
                                                  self.device)
                    layers.extend(mlp)
                return tk.Sequential(layers)

            def call(self, observations: np.ndarray, **kwargs):
                with tf.device(self.device):
                    tensor_observation = tf.convert_to_tensor(observations, dtype=tf.float32)
                    return {'state': self.model(tensor_observation)}



  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.mindspore.representations import *


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

