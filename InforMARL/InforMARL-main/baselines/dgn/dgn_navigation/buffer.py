# DGN+ATOC buffer
import numpy as np


class ATOCReplayBuffer(object):
    def __init__(self, buffer_size, obs_space, n_action, n_ant):
        self.buffer_size = buffer_size
        self.n_ant = n_ant
        self.pointer = 0
        self.len = 0
        self.actions = np.zeros((self.buffer_size, self.n_ant), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size, n_ant))
        self.dones = np.zeros((self.buffer_size, 1))
        self.obs = np.zeros((self.buffer_size, self.n_ant, obs_space))
        self.next_obs = np.zeros((self.buffer_size, self.n_ant, obs_space))
        self.matrix = np.zeros((self.buffer_size, self.n_ant, self.n_ant))
        self.next_matrix = np.zeros((self.buffer_size, self.n_ant, self.n_ant))

    def getBatch(self, batch_size):
        index = np.random.choice(self.len, batch_size, replace=False)
        return (
            self.obs[index],
            self.actions[index],
            self.rewards[index],
            self.next_obs[index],
            self.matrix[index],
            self.next_matrix[index],
            self.dones[index],
        )

    def add(self, obs, action, reward, next_obs, matrix, next_matrix, done):
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_obs[self.pointer] = next_obs
        self.matrix[self.pointer] = matrix
        self.next_matrix[self.pointer] = next_matrix
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)


# DGN buffer
from collections import deque
import random


class DGNReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, obs, action, reward, new_obs, matrix, next_matrix, done):
        experience = (obs, action, reward, new_obs, matrix, next_matrix, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
