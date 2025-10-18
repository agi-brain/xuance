import numpy as np


class ValueNorm:
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(ValueNorm, self).__init__()
        self.input_shapes = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = np.zeros(input_shape)
        self.running_mean_sq = np.zeros(input_shape)
        self.debiasing_term = np.zeros(1, dtype=np.float32)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean = np.zeros(self.input_shapes)
        self.running_mean_sq = np.zeros(self.input_shapes)
        self.debiasing_term = np.zeros(1, dtype=np.float32)

    def running_mean_var(self):
        debiased_mean = self.running_mean / np.clip(self.debiasing_term, self.epsilon, np.inf)
        debiased_mean_sq = self.running_mean_sq / np.clip(self.debiasing_term, self.epsilon, np.inf)
        debiased_var = np.clip(debiased_mean_sq - debiased_mean ** 2, 1e-2, np.inf)
        return debiased_mean, debiased_var

    def update(self, input_vector):
        input_vector = input_vector.numpy()
        batch_mean = input_vector.mean(axis=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(axis=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean = self.running_mean.__mul__(weight).__add__(batch_mean * (1.0 - weight))
        self.running_mean_sq = self.running_mean_sq.__mul__(weight).__add__(batch_sq_mean * (1.0 - weight))
        self.debiasing_term = self.debiasing_term.__mul__(weight).__add__(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / np.sqrt(var)[(None,) * self.norm_axes]

        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        input_vector = input_vector  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = input_vector * np.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        return out
