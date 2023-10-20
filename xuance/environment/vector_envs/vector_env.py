from abc import ABC, abstractmethod
import numpy as np
import cv2


# referenced from openai/baselines
class AlreadySteppingError(Exception):
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


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


class VecEnv(ABC):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode):
        raise NotImplementedError

    def close(self):
        if self.closed == True:
            return
        self.close_extras()
        self.closed = True
