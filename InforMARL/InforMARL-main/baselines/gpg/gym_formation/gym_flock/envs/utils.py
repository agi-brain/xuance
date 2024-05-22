import numpy as np
import re


# N - number of drones
# dist - dist between drones on circumference, 0.5 < 0.75 keeps things interesting
def circle_helper(N, dist):
    r = dist * N / 2 / np.pi
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).reshape((N, 1))
    # angles2 = np.pi - angles
    return r * np.hstack((np.cos(angles), np.sin(angles))), -0.5 * np.hstack(
        (np.cos(angles), -0.5 * np.sin(angles))
    )


def circle(N):
    if N <= 20:
        return circle_helper(N, 0.5)
    else:
        smalln = int(N * 2.0 / 5.0)
        circle1, v1 = circle_helper(smalln, 0.5)
        circle2, v2 = circle_helper(N - smalln, 0.5)
        return np.vstack((circle1, circle2)), np.vstack((v1, v2))


def grid(N, side=5):
    side2 = int(N / side)
    xs = np.arange(0, side) - side / 2.0
    ys = np.arange(0, side2) - side2 / 2.0
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape((N, 1))
    ys = ys.reshape((N, 1))
    return 0.6 * np.hstack((xs, ys))


def twoflocks(N):
    half_n = int(N / 2)
    grid1 = grid(half_n)
    delta = 6
    grid2 = grid1.copy() + np.array([0, delta / 2]).reshape((1, 2))
    grid1 = grid1 + np.array([0, -delta / 2]).reshape((1, 2))

    vels1 = np.tile(np.array([-1.0, delta]).reshape((1, 2)), (half_n, 1))
    vels2 = np.tile(np.array([1.0, -delta]).reshape((1, 2)), (half_n, 1))

    grids = np.vstack((grid1, grid2))
    velss = 0.05 * np.vstack((vels1, vels2))

    return grids, velss


def parse_settings(fname):
    names = []
    homes = []
    for line in open(fname):
        for n in re.findall(r"\"(.+?)\": {", line):
            if n != "Vehicles":
                names.append(n)
        p = re.findall(
            r'"X": ([-+]?\d*\.*\d+), "Y": ([-+]?\d*\.*\d+), "Z": ([-+]?\d*\.*\d+)', line
        )
        if p:
            homes.append(
                np.array([float(p[0][0]), float(p[0][1]), float(p[0][2])]).reshape(
                    (1, 3)
                )
            )
    return names, np.concatenate(homes, axis=0)
