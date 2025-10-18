import random
import numpy as np
from xuance.tensorflow import tf, tk, Module, Tensor


def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
    lr = initial_lr * (1 - step / float(total_steps))
    if lr < end_factor * initial_lr:
        lr = end_factor * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_seed(seed):
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# def get_flat_grad(y: Tensor, model: Module) -> Tensor:
#     grads = torch.autograd.grad(y, model.parameters())
#     return torch.cat([grad.reshape(-1) for grad in grads])


def get_flat_params(model: Module) -> Tensor:
    params = model.parameters()
    return tf.concat([param.reshape(-1) for param in params])


def assign_from_flat_grads(flat_grads: Tensor, model: Module) -> Module:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


def assign_from_flat_params(flat_params: Tensor, model: Module) -> Module:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


class MyLinearLR(tk.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, start_factor, end_factor, total_iters):
        self.initial_learning_rate = initial_learning_rate
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.learning_rate = self.initial_learning_rate
        self.delta_factor = (end_factor - start_factor) * self.initial_learning_rate / self.total_iters

    def __call__(self, step):
        self.learning_rate += self.delta_factor
        return self.learning_rate

