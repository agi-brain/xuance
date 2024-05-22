import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import scipy.linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels


class LQREnv(gym.Env):
    def __init__(self):
        config_file = path.join(path.dirname(__file__), "params_lqr.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config["lqr"]

        self.filter_len = int(config["filter_length"])
        self.n_nodes = int(config["network_size"])
        self.dt = float(config["sampling_dt"])
        self.x_range = 7.0
        self.x_max = float(config["xmax"])
        self.var = float(config["system_variance"])
        self.degree = int(config["degree"])
        self.b_scale = float(config["b_scale"])
        self.alpha = float(config["alpha"])

        # generate node locations
        node_loc = self.alpha * np.random.uniform(0, 1.0, size=(self.n_nodes, 2))

        # generate linear system and geometric network
        a_sys = pairwise_kernels(node_loc, metric="rbf")
        np.fill_diagonal(a_sys, 0)
        neigh = NearestNeighbors(n_neighbors=self.degree)
        neigh.fit(node_loc)

        a_net = a_sys * np.array(neigh.kneighbors_graph(mode="connectivity").todense())

        # discretize system given dt
        a_net = a_net / max(np.abs(np.linalg.eigvals(a_net)))

        a_expm = scipy.linalg.expm(self.dt * a_sys)
        b_sys = (np.linalg.inv(a_sys).dot(a_expm - np.eye(self.n_nodes))).dot(
            self.b_scale * np.eye(self.n_nodes)
        )

        # simplified since A is symmetric:
        q_sys = np.linalg.inv(2 * a_sys).dot(
            scipy.linalg.expm(self.dt * 2.0 * a_sys) - np.eye(self.n_nodes)
        )
        # q_sys is ALMOST symmetric within 1e-16
        q_sys = (q_sys + q_sys.T) / 2.0

        self.a_net = a_net
        self.a_sys = a_expm
        self.b_sys = b_sys
        self.q_sys = q_sys
        self.r_sys = (
            self.dt * np.eye(self.n_nodes) * (self.b_scale**2)
        )  # TODO describe in paper
        self.cov = q_sys * self.var
        self.std_dev = np.sqrt(self.cov[0, 0])

        self.a_net_nan = self.a_net.reshape((self.n_nodes, self.n_nodes, 1))
        self.a_net_nan[self.a_net_nan == 0] = np.nan

        # TODO - tune these to be reasonable
        self.max_u = 40
        self.max_z = 200

        self.action_space = spaces.Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-self.max_z, high=self.max_z, shape=(self.filter_len,), dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, ut):
        xt = self.x
        xt.shape = (self.n_nodes, 1)
        ut.shape = (self.n_nodes, 1)
        xt1 = (
            self.a_sys.dot(xt)
            + self.b_sys.dot(ut)
            + np.random.normal(0, self.std_dev, (self.n_nodes, 1))
        )
        cost = self.instant_cost(xt, ut)

        self.x = xt1
        self.x_agg = self.aggregate(self.x, self.x_agg)

        return self._get_obs(), -cost, False, {}

    def instant_cost(self, xt, ut):  # sum of differences in velocities
        xt.shape = (self.n_nodes, 1)
        ut.shape = (self.n_nodes, 1)
        cost = xt.T.dot(self.q_sys).dot(xt) + ut.T.dot(self.r_sys).dot(ut)
        return cost

    def _get_obs(self):
        reshaped = self.x_agg.reshape((self.n_nodes, self.filter_len))
        return np.clip(reshaped, a_min=-self.max_z, a_max=self.max_z)

    def reset(self):
        self.x = np.random.uniform(
            low=-self.x_max, high=self.x_max, size=(self.n_nodes,)
        )
        self.x_agg = np.zeros((self.n_nodes, self.filter_len))
        self.x_agg = self.aggregate(self.x, self.x_agg)
        return self._get_obs()

    def close(self):
        pass

    def aggregate(self, xt, x_agg):
        """
        Perform aggegration operation
        Args:
            x_agg (): Last time step's aggregated info
            xt (): Current state of all agents

        Returns:
            Aggregated state values
        """
        # get rid of oldest forwarded information
        last_agg = np.array(x_agg[:, :-1]).reshape(
            (self.n_nodes, 1, self.filter_len - 1)
        )

        # get forwarded information from neighbors
        features = np.nansum(last_agg * self.a_net_nan, axis=0).reshape(
            (self.n_nodes, self.filter_len - 1)
        )
        return np.hstack((xt.reshape(self.n_nodes, 1), features))
