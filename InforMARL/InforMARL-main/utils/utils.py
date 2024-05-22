import numpy as np
import argparse
import requests
import inspect
import functools


def store_args(method):
    """
    https://stackoverflow.com/questions/6760536/python-iterating-through-constructors-arguments
    https://github.com/openai/baselines/blob/master/baselines/her/util.py
    Stores provided method args as instance attributes.
    Usage:
    ------
    class A:
        @store_args
        def __init__(self, a, b, c=3, d=4, e=5):
            pass
    a = A(1,2)
    print(a.a, a.b, a.c, a.d, a.e)
    >>> 1 2 3 4 5
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def print_dash(num_dash: int = 50):
    """
    Print "______________________"
    for num_dash times
    """
    print("_" * num_dash)


def print_box(string, num_dash: int = 50):
    """
    print the given string as:
    _____________________
    string
    _____________________
    """
    print_dash(num_dash)
    print(string)
    print_dash(num_dash)


def print_args(args: argparse.Namespace):
    """
    Print the args in a pretty table
    """
    box_dist = 50
    print_dash(box_dist)
    half_len = int((box_dist - len("Arguments") - 5) / 2)
    print("||" + " " * half_len + "Arguments" + " " * half_len + " ||")
    print_dash(box_dist)
    for k, v in vars(args).items():
        len_line = len(f"{k}: {str(v)}")
        print("|| " + f"{k}: {str(v)}" + " " * (box_dist - len_line - 5) + "||")
    print_dash(box_dist)


def connected_to_internet(url: str = "http://www.google.com/", timeout: int = 5):
    """
    Check if system is connected to the internet
    Used when running code on MIT Supercloud
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("No internet connection available.")
    return False


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c
