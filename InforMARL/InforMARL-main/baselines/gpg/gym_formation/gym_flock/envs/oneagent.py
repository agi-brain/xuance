import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from sklearn.neighbors import NearestNeighbors
import itertools
import random
import pdb

font = {"family": "sans-serif", "weight": "bold", "size": 14}


class OneAgentEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self):
        # pdb.set_trace()
        self.n_agents = 1
        self.max_accel = 0.3
        self.n_features = 4
        self.v_max = float(2)
        self.v_bias = self.v_max
        self.gain = 0.5
        self.dt = 0.1
        self.action_space = spaces.Box(
            low=-self.max_accel,
            high=self.max_accel,
            shape=(2 * self.n_agents,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.Inf, high=np.Inf, shape=(self.n_features,), dtype=np.float32
        )
        self.fig = None
        self.r_max = 4
        self.nu = 2

    def reset(self):
        self.counter = 0
        self.feats = np.zeros((self.n_agents, 2))
        x = np.zeros((self.n_agents, self.n_features))
        self.goal_x1 = np.random.uniform(3.5, 4.5)
        self.goal_y1 = np.random.uniform(3.5, 4.5)
        # xpoints = np.array((0))
        # ypoints = np.array((0))

        # self.goal_x1 = np.array((2))
        # self.goal_y1 = np.array((2))
        xpoints = np.random.uniform(-1, 1)
        ypoints = np.random.uniform(-1, 1)
        # xpoints = np.array((2))
        # ypoints = np.array((2))

        self.start_xpoint = xpoints
        self.start_ypoint = ypoints
        # xpoints = np.array((2.0))
        # ypoints = np.array((2.0))

        x[:, 0] = xpoints
        x[:, 1] = ypoints

        bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        # x[2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0]
        # x[3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1]

        x[:, 2] = np.array((self.goal_x1))
        x[:, 3] = np.array((self.goal_y1))
        self.x = x
        return self._get_obs()

    def _get_obs(self):
        self.feats[:, 0] = self.x[:, 0] - self.x[:, 2]
        self.feats[:, 1] = self.x[:, 1] - self.x[:, 3]

        # return (state_values, state_network)
        return self.feats

    def step(self, action):
        self.u = np.reshape(action, (self.n_agents, self.nu))
        self.counter += 1
        # x velocity
        # self.x[2] = self.x[2] + self.gain * self.u[0] * self.dt #+ np.random.normal(0, self.std_dev, (self.n_agents,))
        # y velocity
        # self.x[3] = self.x[3] + self.gain * self.u[1] * self.dt #+ np.random.normal(0, self.std_dev, (self.n_agents,))

        # if self.dynamic:
        # x position
        self.x[:, 0] = self.x[:, 0] + self.u[:, 0]
        # y position
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1]

        reward = -(
            ((self.x[:, 0] - self.x[:, 2]) ** 2 + (self.x[:, 1] - self.x[:, 3]) ** 2)
            ** 0.5
        )

        done = False

        diffs_x = np.abs(self.x[:, 0] - self.goal_x1)
        diffs_y = np.abs(self.x[:, 1] - self.goal_y1)

        if self.counter > 400:
            done = True
        if np.all(diffs_x) < 0.2 and np.all(diffs_y) < 0.2:
            done = True
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            (line1,) = ax.plot(
                self.x[:, 0], self.x[:, 1], "bo"
            )  # Returns a tuple of line objects, thus the comma
            # line2, = ax.plot(self.x[:, 4], self.x[:, 5], 'go')
            ax.plot(self.goal_x1, self.goal_y1, "kx")
            ax.plot(self.start_xpoint, self.start_ypoint, "bx")
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title("GNN Controller")
            self.fig = fig
            self.line1 = line1
            # self.line2 = line2

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        # self.line2.set_xdata(self.x[:, 4])
        # self.line2.set_ydata(self.x[:, 5])

        # self.line1.set_xdata(self.x[:, 2])
        # self.line1.set_ydata(self.x[:, 3])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
