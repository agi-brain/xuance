""" some utilities """

import collections
import os


def rec_round(x, ndigits=2):
    """round x recursively

    Parameters
    ----------
    x: float, int, list, list of list, ...
        variable to round, support many types
    ndigits: int
        precision in decimal digits
    """
    if isinstance(x, collections.abc.Iterable):
        return [rec_round(item, ndigits) for item in x]
    return round(x, ndigits)


def download_file(filename, url):
    """download url to filename"""
    print(f"Download {filename} from {url}...")

    ret = os.system(f"wget -O {filename} '{url}'")

    if ret != 0:
        print("ERROR: wget fails!")
        print(
            "If you are an OSX user, you can install wget by 'brew install wget' and retry."
        )
        exit(-1)
    else:
        print("download done!")


def download_model(url):
    """download model from url"""
    name = url.split("/")[-1]
    name = os.path.join("data", name)
    download_file(name, url)

    def do_commond(cmd):
        print(cmd)
        os.system(cmd)

    do_commond("tar xzf %s -C data" % name)
    do_commond("rm %s" % name)


def check_model(name):
    """check whether a model is downloaded"""
    infos = {
        "against": (
            ("data/battle_model/battle/tfdqn_0.index",),
            "https://raw.githubusercontent.com/merrymercy/merrymercy.github.io/master/_data/magent/against-0.tar.gz",
        ),
        "battle-game": (
            (
                "data/battle_model/trusty-battle-game-l/tfdqn_0.index",
                "data/battle_model/trusty-battle-game-r/tfdqn_0.index",
            ),
            "https://raw.githubusercontent.com/merrymercy/merrymercy.github.io/master/_data/magent/battle_model.tar.gz",
        ),
        "arrange": (
            ("data/arrange_model/arrange/tfdqn_10.index",),
            "https://raw.githubusercontent.com/merrymercy/merrymercy.github.io/master/_data/magent/arrange_game.tar.gz",
        ),
    }

    if name not in infos:
        raise RuntimeError("Unknown model name")

    info = infos[name]
    missing = False
    for check in info[0]:
        if not os.path.exists(check):
            missing = True
    if missing:
        download_model(info[1])


class FontProvider:
    """provide pixel font"""

    def __init__(self, filename):
        data = []
        # read raw
        with open(filename) as fin:
            for line in fin.readlines():
                char = []
                for x in line.split(","):
                    char.append(eval(x))
                data.append(char)

        height = 8
        width = 8

        # expand bit compress
        expand_data = []
        for char in data:
            expand_char = [[0 for _ in range(width)] for _ in range(height)]
            for i in range(width):
                for j in range(height):
                    set = char[i] & (1 << j)
                    if set:
                        expand_char[i][j] = 1
            expand_data.append(expand_char)

        self.data = expand_data
        self.width = width
        self.height = height

    def get(self, i):
        if isinstance(i, int):
            return self.data[i]
        else:
            return self.data[ord(i)]
