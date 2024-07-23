import random
import numpy as np
from .distributions import CategoricalDistribution, DiagGaussianDistribution
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


def split_distributions(distribution):
    return_list = []
    if isinstance(distribution, CategoricalDistribution):
        shape = distribution.logits.shape
        logits = tf.reshape(distribution.logits, [-1, shape[-1]])
        for logit in logits:
            dist = CategoricalDistribution(logits.shape[-1])
            dist.set_param(logits=tf.stop_gradient(tf.expand_dims(logit, 0)))
            return_list.append(dist)
    elif isinstance(distribution, DiagGaussianDistribution):
        shape = distribution.mu.shape
        means = tf.reshape(distribution.mu, [-1, shape[-1]])
        std = distribution.std
        for mu in means:
            dist = DiagGaussianDistribution(shape[-1])
            dist.set_param(mu, std)
            return_list.append(dist)
    else:
        raise NotImplementedError
    return np.array(return_list).reshape(shape[:-1])


def merge_distributions(distribution_list):
    if isinstance(distribution_list[0], CategoricalDistribution):
        logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        dist.set_param(tf.stop_gradient(logits))
        return dist
    elif isinstance(distribution_list[0], DiagGaussianDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        mu = tf.concat([dist.mu for dist in distribution_list], axis=0)
        std = tf.concat([dist.std for dist in distribution_list], axis=0)
        action_dim = distribution_list[0].mu.shape[-1]
        dist = DiagGaussianDistribution(action_dim)
        mu = tf.reshape(mu, shape + (action_dim,))
        std = tf.reshape(std, shape + (action_dim,))
        dist.set_param(mu, std)
        return dist
    elif isinstance(distribution_list[0, 0], CategoricalDistribution):
        shape = distribution_list.shape
        distribution_list = distribution_list.reshape([-1])
        logits = tf.concat([dist.logits for dist in distribution_list], axis=0)
        action_dim = logits.shape[-1]
        dist = CategoricalDistribution(action_dim)
        logits = tf.reshape(logits, shape + (action_dim, ))
        dist.set_param(tf.stop_gradient(logits))
        return dist
    else:
        pass


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

