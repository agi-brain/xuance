import numpy as np
from typing import Dict


def space2shape(observation_space):
    """Convert gym.space variable to shape
    Args:
        observation_space: the space variable with type of gym.Space.

    Returns:
        The shape of the observation_space.
    """
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape


def combined_shape(length: int, shape=None):
    """Expand the original shape.

    Args:
        length (int): The length of the first dimension to prepend.
        shape (int, list, tuple, or None): The target shape to be expanded.
                                           It can be an integer, a sequence, or None.

    Returns:
        tuple: A new shape expanded from the input shape.

    Examples:
        >>> length = 2
        >>> shape_1 = None
        >>> shape_2 = 3
        >>> shape_3 = [4, 5]
        >>> combined(length, shape_1)
        (2, )
        >>> combined(length, shape_2)
        (2, 3)
        >>> combined(length, shape_3)
        (2, 4, 5)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
