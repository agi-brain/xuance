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
