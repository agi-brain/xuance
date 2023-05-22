import numpy as np
from .distributions import CategoricalDistribution, DiagGaussianDistribution
import tensorflow as tf
import tensorflow.keras as tk


# def get_flat_grad(y: tf.Tensor, model: tk.Model) -> tf.Tensor:
#     grads = torch.autograd.grad(y, model.parameters())
#     return torch.cat([grad.reshape(-1) for grad in grads])


def get_flat_params(model: tk.Model) -> tf.Tensor:
    params = model.parameters()
    return tf.concat([param.reshape(-1) for param in params])


def assign_from_flat_grads(flat_grads: tf.Tensor, model: tk.Model) -> tk.Model:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad.copy_(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


def assign_from_flat_params(flat_params: tf.Tensor, model: tk.Model) -> tk.Model:
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
            dist.set_param(tf.stop_gradient(tf.expand_dims(logit, 0)))
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




