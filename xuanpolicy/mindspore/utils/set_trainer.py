import mindspore as ms
import mindspore.nn as nn


clip_grad = ms.ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = ms.ops.dtype(grad)
    if clip_type == 0:
        new_grad = ms.ops.clip_by_value(grad, ms.ops.cast(ms.ops.tuple_to_array((-clip_value,)), dt),
                                   ms.ops.cast(ms.ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ms.ops.cast(ms.ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainOneStepCellWithGradClip(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True,
                 clip_value=None, clip_type=None):
        super(TrainOneStepCellWithGradClip, self).__init__(network, optimizer, sens)
        self.cast = ms.ops.Cast()
        self.hyper_map = ms.ops.HyperMap()
        self.enable_clip_grad = enable_clip_grad
        self.clip_value = clip_value
        self.clip_type = clip_type

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        grads = self.grad(self.network, weights)(*inputs, self.cast(ms.ops.tuple_to_array((self.sens,)), ms.float32))
        if self.enable_clip_grad:
            grads = self.hyper_map(ms.ops.partial(clip_grad, self.clip_type, self.clip_value), grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss