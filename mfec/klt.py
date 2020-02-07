#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k_exp, k_act, state_dim, M, explore, time_sig, seed):
        self.buffer_size = buffer_size
        self.time_sig = time_sig
        self.explore = explore
        self.dist_sig = explore
        self.buffers = tuple(
            [ActionBuffer(n=a,
                          capacity=self.buffer_size,
                          state_dim=state_dim,
                          M=M,
                          seed=seed,
                          time_sig=time_sig,
                          ) for a in actions])
        self.k_exp = k_exp
        self.k_act = k_act

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
            return 1e6

        k = min(self.k_act, n)
        neighbors, dists = buffer.find_neighbors(state, k)
        neighbors = neighbors[0]
        dists = np.sqrt(dists[0]) + 0.01

        v_over_time = [buffer.values_list[n][0] for n in neighbors]

        if self.dist_sig =="mean":
            weighted_reward = np.mean(v_over_time)
        elif self.dist_sig=="best fixed":
            w = self.laplace(dists, 110)+1e-6
            sum_w = np.sum(w)
            weighted_reward = np.dot(v_over_time, w) / sum_w
        elif self.dist_sig == "inverse":
            w = np.divide(1, dists+1e-6)
            sum_w = np.sum(w)
            weighted_reward = np.dot(v_over_time, w) / sum_w
        # times_lists = [buffer.times_list[n] for n in neighbors]

        # v_over_time = []
        # for times, values in zip(times_lists, values_lists):
        #    w_t = self.laplace(time - np.asarray(times), self.time_sig) + 0.01
        #    val = np.dot(values, w_t)
        #    v_over_time.append(val)

        return weighted_reward

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
    def __init__(self, n, capacity, state_dim, M, time_sig, seed):
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
        self.a = time_sig

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
                self.values_list[idx][0] = (1 - self.a) * self.values_list[idx][0] + self.a * value
                # self.times_list[idx].append(time)
                return

        # Otherwise it's new
        self._tree.add_items(state)
        self.values_list.append([value])
        # self.times_list.append([time])
        self.states.append(state)
        self.length += 1

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        """number of samples in the buffer - not number visited."""
        return self.length
