import numpy as np
from xuance.mindspore import ms, ops, Module, Tensor


class ValueNorm(Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = ms.Parameter(ops.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = ms.Parameter(ops.zeros(input_shape), requires_grad=False)
        self.debiasing_term = ms.Parameter(Tensor(0.0), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.set_data(ops.zeros(self.running_mean.shape))
        self.running_mean_sq.set_data(ops.zeros(self.running_mean_sq.shape))
        self.debiasing_term.set_data(ops.zeros(self.debiasing_term.shape))

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = Tensor(input_vector)

        batch_mean = input_vector.mean(axis=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(axis=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul(weight).add(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul(weight).add(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul(weight).add(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = Tensor(input_vector)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / ops.sqrt(var)[(None,) * self.norm_axes]

        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        input_vector = Tensor(input_vector)

        mean, var = self.running_mean_var()
        out = input_vector * ops.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        out = out.asnumpy()

        return out
