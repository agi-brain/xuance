MLP-based
=====================================

多层感知器是一种最简单的深层神经网络模型，用于处理向量输入，
用户可根据各自需要实例化多层感知器模块，
其定义位于./xuance_torch/utils/layers.py和./xuance_ms/utils/layers.py文件中，类名称为mlp_block。
实例化该类需指定输入维度大小（input_dim），输出维度大小（output_dim），归一化方法（normalize），
激活函数选择（activation），初始化方法（initialize）。
在pytorch下实现还需指定设备类型（device），以确定模型在CPU上运行还是GPU。

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuanpolicy.torch.representations.mlp.Basic_Identical(input_shape, device)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence[int]
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.representations.mlp.Basic_Identical.forward(observations)

    Calculate feature representation of the input observations.

    :param observations: The observation of current step.
    :type observations: numpy.ndarray
    :return: The features output by the representation model.
    :rtype: dict

.. py:class:: 
    xuanpolicy.torch.representations.mlp.Basic_MLP(input_shape, device)

    :param input_shape: The shape of the inputs.
    :type input_shape: Sequence[int]
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuanpolicy.torch.representations.mlp.Basic_MLP._create_network()

    Create the multi-layer perceptron netowrks.

    :return: The neural network module.
    :rtype: nn.Module

.. py:function:: 
    xuanpolicy.torch.representations.mlp.Basic_MLP.forward(observations)

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

    .. code-block:: python3

  .. group-tab:: MindSpore

    .. code-block:: python3
