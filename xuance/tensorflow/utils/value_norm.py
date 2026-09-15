import numpy as np
import tensorflow as tf


class ValueNorm:
    """ normalizer a vector of observations - across the first norm_axes dimensions"""

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

    def update_tensor(self,
                      running_mean: tf.Tensor,
                      running_mean_sq: tf.Tensor,
                      debiasing_term: tf.Tensor):
        self.running_mean = running_mean.numpy()
        self.running_mean_sq = running_mean_sq.numpy()
        self.debiasing_term = debiasing_term.numpy()

    def reset_parameters(self):
        self.running_mean = np.zeros(self.input_shapes)
        self.running_mean_sq = np.zeros(self.input_shapes)
        self.debiasing_term = np.zeros(1, dtype=np.float32)

    def running_mean_var(self):
        debiased_mean = self.running_mean / np.clip(self.debiasing_term, self.epsilon, np.inf)
        debiased_mean_sq = self.running_mean_sq / np.clip(self.debiasing_term, self.epsilon, np.inf)
        debiased_var = np.clip(debiased_mean_sq - debiased_mean ** 2, 1e-2, np.inf)
        return debiased_mean, debiased_var

    def update(self, input_vector,
               running_mean=None,
               running_mean_sq=None,
               debiasing_term=None):
        input_vector = tf.cast(input_vector, tf.float32)
        axes = tuple(range(self.norm_axes))
        batch_mean = tf.reduce_mean(input_vector, axis=axes)
        batch_sq_mean = tf.reduce_mean(tf.square(input_vector), axis=axes)

        if self.per_element_update:
            shape = tf.shape(input_vector)
            batch_size = tf.reduce_prod(shape[:self.norm_axes])
            weight = tf.pow(tf.cast(self.beta, tf.float32), tf.cast(batch_size, tf.float32))
        else:
            weight = tf.cast(self.beta, tf.float32)

        running_mean_new = running_mean * weight + batch_mean * (1.0 - weight)
        running_mean_sq_new = running_mean_sq * weight + batch_sq_mean * (1.0 - weight)
        debiasing_term_new = debiasing_term * weight + (1.0 - weight)

        return running_mean_new, running_mean_sq_new, debiasing_term_new

    def normalize(self, input_vector,
                  running_mean: tf.Tensor = None,
                  running_mean_sq: tf.Tensor = None,
                  debiasing_term: tf.Tensor = None):
        # Make sure input is float32
        input_vector = tf.cast(input_vector, tf.float32)  # not elegant, but works in most cases

        debiasing_term = tf.clip_by_value(
            debiasing_term,
            clip_value_min=self.epsilon,
            clip_value_max=tf.float32.max,
        )
        debiased_mean = running_mean / debiasing_term
        debiased_mean_sq = running_mean_sq / debiasing_term
        debiased_var = tf.maximum(debiased_mean_sq - tf.square(debiased_mean), 1e-2)
        mean, var = debiased_mean, debiased_var
        out = (input_vector - mean) / tf.sqrt(var)

        return out

    def denormalize(self, input_vector):
        """ Transform normalizerd data back into original distribution """
        input_vector = input_vector  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = input_vector * np.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        return out
