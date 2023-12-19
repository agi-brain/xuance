from mpi4py import MPI
from typing import Union, Sequence
import numpy as np


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n + 1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]


def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


class RunningMeanStd(object):
    def __init__(self,
                 shape: Union[Sequence[int], dict],
                 epsilon=1e-4,
                 comm=None,
                 use_mpi=False):
        self.shape = shape
        if isinstance(shape, dict):
            self.mean = {key: np.zeros(shape[key], np.float32) for key in shape.keys()}
            self.var = {key: np.ones(shape[key], np.float32) for key in shape.keys()}
            self.count = {key: epsilon for key in shape.keys()}
        else:
            self.mean = np.zeros(shape, np.float32)
            self.var = np.ones(shape, np.float32)
            self.count = epsilon
        self.use_mpi = use_mpi
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

    @property
    def std(self):
        if isinstance(self.shape, dict):
            return {key: np.sqrt(self.var[key]) for key in self.shape.keys()}
        else:
            return np.sqrt(self.var)

    def update(self, x):
        if isinstance(x, dict):
            batch_means = {}
            batch_vars = {}
            batch_counts = {}
            for key in self.shape.keys():
                if self.use_mpi:
                    batch_mean, batch_std, batch_count = mpi_moments(x[key], axis=0, comm=self.comm)
                else:
                    batch_mean, batch_std, batch_count = np.mean(x[key], axis=0), np.std(x[key], axis=0), x[key].shape[
                        0]
                batch_means[key] = batch_mean
                batch_vars[key] = np.square(batch_std)
                batch_counts[key] = batch_count
            self.update_from_moments(batch_means, batch_vars, batch_counts)
        else:
            if self.use_mpi:
                batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
            else:
                batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
            batch_var = np.square(batch_std)
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if isinstance(batch_mean, dict):
            for key in self.shape:
                delta = batch_mean[key] - self.mean[key]
                tot_count = self.count[key] + batch_count[key]
                new_mean = self.mean[key] + delta * batch_count[key] / tot_count
                m_a = self.var[key] * (self.count[key])
                m_b = batch_var[key] * (batch_count[key])
                M2 = m_a + m_b + np.square(delta) * self.count[key] * batch_count[key] / (
                            self.count[key] + batch_count[key])
                new_var = M2 / (self.count[key] + batch_count[key])
                new_count = batch_count[key] + self.count[key]
                self.mean[key] = new_mean
                self.var[key] = new_var
                self.count[key] = new_count
        else:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)
            new_count = batch_count + self.count
            self.mean = new_mean
            self.var = new_var
            self.count = new_count
