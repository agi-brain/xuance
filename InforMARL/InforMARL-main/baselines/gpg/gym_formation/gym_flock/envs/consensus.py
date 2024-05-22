import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {"family": "sans-serif", "weight": "bold", "size": 14}


class ConsensusEnv(gym.Env):
    def __init__(self):
        config_file = path.join(path.dirname(__file__), "params_consensus.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config["flock"]

        self.fig = None
        self.line1 = None
        self.filter_len = int(config["filter_length"])
        self.nx_system = 1
        self.n_nodes = int(config["network_size"])
        self.comm_radius = float(config["comm_radius"])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = 0.001  # float(config['system_dt'])
        self.v_max = 10.0  # float(config['max_vel_init'])
        self.v_bias = 6.0  # 5 * self.v_max  # 0.5 * self.v_max
        self.r_max = 50.0  # float(config['max_rad_init'])
        self.std_dev = float(config["std_dev"]) * self.dt

        self.pooling = [np.nanmean]
        self.n_pools = len(self.pooling)

        # number of features and outputs

        self.nx = 2
        self.nu = 1
        self.n_features = self.nx * self.filter_len  # int(config['N_features'])

        self.x_agg = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        self.x = np.zeros((self.n_nodes, self.nx_system))
        self.u = np.zeros((self.n_nodes, self.nu))
        self.mean_val = np.zeros((self.n_nodes, self.nu))
        self.init_val = np.zeros((self.n_nodes, self.nu))

        # TODO
        self.max_accel = 10.0
        self.max_z = 200

        self.action_space = spaces.Box(
            low=-self.max_accel,
            high=self.max_accel,
            shape=(self.nu * self.n_nodes,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-self.max_z,
            high=self.max_z,
            shape=(self.n_features * self.n_nodes,),
            dtype=np.float32,
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.u = u.reshape((self.n_nodes, self.nu))
        self.u = np.clip(self.u, a_min=-self.max_accel, a_max=self.max_accel)
        self.x = self.x + self.u * self.dt
        self.x_agg = self.aggregate(self.x, self.x_agg)
        self.u = u
        return (self._get_obs(), self.cost_list()), self.instant_cost(), False, {}

    def instant_cost(self):  # sum of differences in velocities
        s_costs = -1.0 * np.square(
            self.x - self.mean_val
        )  # - np.sum(np.square(self.u)) * 0.001
        # s_costs =  np.exp(-0.1 * np.square(self.x - self.mean_val)) #- np.sum(np.square(self.u)) * 0.001
        # s_costs = -1.0 * np.log(np.square(self.x - self.mean_val) + 0.01)
        return np.sum(s_costs)  # + np.sum(np.square(self.u)) # todo add an action cost

    def cost_list(self):  # sum of differences in velocities
        # s_costs = -1.0 * np.square(self.x - self.mean_val).flatten() #- np.square(self.u).flatten() * 0.001
        # s_costs =  np.exp(-0.1 * np.square(self.x - self.mean_val).flatten()) #- np.square(self.u).flatten() * 0.001
        # s_costs = -1.0 * np.log(np.square(self.x - self.mean_val) + 0.01)
        # s_costs =  (self.mean_val - self.x).flatten() * self.u.flatten() #(self.mean_val - self.x).flatten()
        s_costs = (self.mean_val - self.x).flatten()
        return s_costs  # + np.sum(np.square(self.u)) # todo add an action cost

    def _get_obs(self):
        reshaped = self.x_agg.reshape((self.n_nodes, self.n_features))
        clipped = np.clip(reshaped, a_min=-self.max_z, a_max=self.max_z)
        return clipped.flatten()  # [self.n_leaders:, :]

    def reset(self):
        x = np.zeros((self.n_nodes, 2))
        degree = 0
        min_dist = 0

        while degree < 2 or min_dist < 0.1:  # < 0.25:  # 0.25:  #0.5: #min_dist < 0.25:
            # randomly initialize the state of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_nodes,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_nodes,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            # compute distances between agents
            x_t_loc = x[:, 0:2]  # x,y location determines connectivity

            a_net = np.sqrt(
                np.sum(
                    np.square(
                        x_t_loc.reshape((self.n_nodes, 1, 2))
                        - x_t_loc.reshape((1, self.n_nodes, 2))
                    ),
                    axis=2,
                )
            )

            # no self loops
            a_net = a_net + 2 * self.comm_radius * np.eye(self.n_nodes)

            # compute minimum distance between agents and degree of network
            min_dist = np.min(np.min(a_net))
            a_net = a_net < self.comm_radius
            degree = np.min(np.sum(a_net.astype(int), axis=1))
        a_net = a_net.astype(float)
        a_net[a_net == 0] = np.nan
        self.a_net = a_net

        ################################
        # bias = (np.floor(np.random.random() * 2)*2 - 1) * self.v_bias
        bias = (np.random.random() * 2 - 1) * self.v_bias
        self.x = (
            np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_nodes, 1))
            + bias
        )
        self.mean_val = np.mean(self.x)
        self.init_val = self.x

        self.x_agg = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        self.x_agg = self.aggregate(self.x, self.x_agg)

        return self._get_obs()

    def close(self):
        pass

    def aggregate(self, xt, x_agg):
        x_features = self.get_x_features(xt)
        for k in range(0, self.n_pools):
            comm_data = self.get_comms(self.get_features(x_agg[:, :, k]))
            x_agg[:, :, k] = np.hstack(
                (x_features, self.get_pool(comm_data, self.pooling[k]))
            )
        return x_agg

    def get_x_features(self, xt):
        return np.hstack((xt, self.init_val))

    def get_features(self, agg):
        return np.tile(
            agg[:, : -self.nx].reshape((self.n_nodes, 1, -1)), (1, self.n_nodes, 1)
        )  # TODO check indexing

    def get_comms(self, mat):
        return mat * self.a_net.reshape(self.n_nodes, self.n_nodes, 1)

    def get_pool(self, mat, func):
        temp_pool = func(mat, axis=1).reshape((self.n_nodes, self.n_features - self.nx))
        temp_pool[np.isnan(temp_pool)] = 0
        return temp_pool

    def controller(self, centralized=True):
        if not centralized:
            comms = self.get_comms(self.get_x_features(self.x))
            temp_pool = np.nanmean(comms, axis=1).reshape((self.n_nodes, 2))
            temp_pool[np.isnan(temp_pool)] = 0
            # print(temp_pool)
            u = temp_pool[:, 0] - self.x.flatten()
        else:
            u = self.mean_val - self.x
        u = u / self.dt
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        return u
