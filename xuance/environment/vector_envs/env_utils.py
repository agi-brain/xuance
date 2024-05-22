import contextlib
import os
from collections import OrderedDict

import gym
import numpy as np


def tile_images(images):
    image_nums = len(images)
    image_shape = images[0].shape
    image_height = image_shape[0]
    image_width = image_shape[1]

    rows = (image_nums - 1) // 4 + 1
    if image_nums >= 4:
        cols = 4
    else:
        cols = image_nums

    try:
        big_img = np.zeros(
            (rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1), image_shape[2]), np.uint8)
    except IndexError:
        big_img = np.zeros((rows * image_height + 10 * (rows - 1), cols * image_width + 10 * (cols - 1)), np.uint8)

    for i in range(image_nums):
        c = i % 4
        r = i // 4
        big_img[10 * r + image_height * r:10 * r + image_height * r + image_height,
        10 * c + image_width * c:10 * c + image_width * c + image_width] = images[i]
    return big_img


def copy_obs_dict(obs):
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_space_info(obs_space):
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        subspaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


# for multi-agent systems
def obs_n_space_info(obs_n_space):
    if isinstance(obs_n_space, gym.spaces.Dict):
        assert isinstance(obs_n_space.spaces, OrderedDict)
        subspaces = obs_n_space.spaces
    elif isinstance(obs_n_space, gym.spaces.Tuple):
        subspaces = {i: obs_n_space.spaces[i] for i in range(len(obs_n_space_info.spaces))}
    elif isinstance(obs_n_space, dict):
        subspaces = {k: obs_n_space[k] for k in obs_n_space.keys()}
    else:
        subspaces = {None: obs_n_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape  # assume the obs_shapes are the same.
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI
    environment variables, MPI will think that the child process is an MPI process just
    like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily
    such as when we are starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])
    return [l__ for l_ in l for l__ in l_]


def flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
