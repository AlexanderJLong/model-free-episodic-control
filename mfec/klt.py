#!/usr/bin/env python3

from collections import deque

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, M, time_sig, seed):
        self.buffer_size = buffer_size
        self.time_sig = time_sig
        self.buffers = tuple(
            [ActionBuffer(n=a,
                          capacity=self.buffer_size,
                          state_dim=state_dim,
                          M=M,
                          seed=seed,
                          ) for a in actions])
        self.k = k

    def gaus(self, x, sig):
        # Goes to 0 in 2xsig
        return np.exp(-np.square(np.divide(x, sig)))

    def laplace(self, x, sig):
        # Goes to 0 in 4xsig
        return np.exp(-np.divide(x, sig))

    def gaus_2d(self, x, y, sig1, sig2):
        return np.exp(-np.square(x / sig1) - np.square(y / sig2))

    def laplace_2d(self, x, y, sig1, sig2):
        return np.exp(-(x / sig1) - (y / sig2))

    def update_normalization(self, maxes, mins):
        for b in self.buffers:
            b.update_normalization(mins=mins, maxes=maxes)

    def estimate(self, state, action, time):
        """Return the estimated value of the given state"""
        buffer = self.buffers[action]

        n = buffer.length
        if n == 0:
            return 1e6, 0.01
        k = min(self.k, n)
        neighbors, dists = buffer.find_neighbors(state, k)
        neighbors = neighbors[0]
        dists = np.sqrt(dists[0])

        # print(dists)
        values_lists = [buffer.values_list[n] for n in neighbors]
        times_lists = [buffer.times_list[n] for n in neighbors]
        counts = np.zeros(k)

        samples = []  # dist, time seperation, value
        for i in range(k):
            times = times_lists[i]
            values = values_lists[i]
            counts[i] = len(values)
            for t, v in zip(times, values):
                samples.append([dists[i], time - t, v])

        samples = np.asarray(samples)

        w = self.laplace_2d(samples[:, 0], samples[:, 1], np.min(dists)+0.01, self.time_sig)+0.01
        w_sum = np.sum(w)
        weighted_reward = np.dot(samples[:, 2], w)/ w_sum

        #b = self.gaus(dists, np.min(dists)+0.01)
        #weighted_count = np.sum(w)

        return weighted_reward, 0

    def update(self, state, action, value, time):
        buffer = self.buffers[action]
        buffer.add(state, value, time)

    def save_indexes(self, save_dir):
        """
        Serialize the index and values. Must be done this way because hnswlib index's cannot be pickled
        """
        for i, buff in enumerate(self.buffers):
            buff._tree.save_index(f"{save_dir}/buff_{i}.bin")

    def load_indexes(self, save_dir):
        """
        Load the index and values. Assumes index has already been initialized
        """
        for i, buff in enumerate(self.buffers):
            buff._tree.load_index(f"{save_dir}/buff_{i}.bin", max_elements=self.buffer_size)

    def plot3d(self, ):
        fig = plt.figure()
        reducer = umap.UMAP(n_neighbors=200, n_components=2)

        fig.set_tight_layout(True)
        rows = 4
        max_val = np.max([max(b.values_list) for b in self.buffers])
        for i, buffer in enumerate(self.buffers):
            ax = fig.add_subplot(rows, len(self.buffers) // rows + 1, i + 1)
            states = np.asarray(buffer.get_states())
            embeddings = reducer.fit_transform(states)
            vals = np.asarray(buffer.values_list)
            ax.scatter(embeddings[:, 1], embeddings[:, 0], c=vals, s=2, vmax=max_val)
            ax.set(title=f"max r={max(vals)}")
        plt.show()
        return


class ActionBuffer:
    def __init__(self, n, capacity, state_dim, M, seed):
        self.id = n
        self.state_dim = state_dim
        self.capacity = capacity
        self.M = M
        self.ef_construction = 500
        self._tree = hnswlib.Index(space="l2", dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity,
                              M=self.M,
                              ef_construction=self.ef_construction,
                              random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.times_list = []
        self.length = 0
        self.states = []
        self.seed = seed

    def __getstate__(self):
        # pickle everything but the hnswlib indexes
        self._tree.save_index(f"saves/index_{self.id}.bin")
        return dict((k, v) for (k, v) in self.__dict__.items() if k != "_tree")

    def __setstate__(self, d):
        self.__dict__ = d
        self._tree = hnswlib.Index(space="l2", dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    @staticmethod
    def safe_divide(a, b):
        """
        If 0 in b, set result in a to 0. This is good in normalizatoin because axis with 0 variance
        should be ignored
        """
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(state, k=k)

    def add(self, state, value, time):
        if self.length != 0:
            idx, dist = self.find_neighbors(state, k=1)
            idx = idx[0][0]
            dist = dist[0][0]
            if dist == 0:
                # existing state
                self.values_list[idx].append(value)
                self.times_list[idx].append(time)
                return

        # Otherwise it's new
        self._tree.add_items(state)
        self.values_list.append([value])
        self.times_list.append([time])
        self.states.append(state)
        self.length += 1

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        """number of samples in the buffer - not number visited."""
        return self.length
