#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, M, seed):
        self.buffer_size = buffer_size
        self.buffers = tuple(
            [ActionBuffer(n=a,
                          capacity=self.buffer_size,
                          state_dim=state_dim,
                          M=M,
                          seed=seed,
                          ) for a in actions])
        self.k = k

    def gaus(self, x, sig):
        return np.exp(-np.square(x / sig) / 2)

    def gaus_2d(self, x, y, sig1, sig2):
        return np.exp(-(np.square(x / sig1) + np.square(y / sig2)))

    def update_normalization(self, mean, std):
        for b in self.buffers:
            b.update_normalization(mean=mean, std=std)

    def estimate(self, state, action):
        """Return the estimated value of the given state"""
        buffer = self.buffers[action]

        n = len(buffer)
        if n == 0:
            return 1e6, 0
        k = min(self.k, n)
        neighbors, dists = buffer.find_neighbors(state, k)
        neighbors = neighbors[0]

        values = [buffer.values_list[n] for n in neighbors]
        weighted_reward = np.mean(values)

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
        for i, buffer in enumerate(self.buffers):
            ax = fig.add_subplot(rows, len(self.buffers) // rows + 1, i + 1)

            states = np.asarray(buffer.get_states())
            embeddings = reducer.fit_transform(states)
            vals = np.asarray(buffer.values_list)
            ax.scatter(embeddings[:, 1], embeddings[:, 0], c=vals, s=1)
            ax.set(xlabel="Vel")
            ax.set(ylabel="Angle")

            ax.set(title=f"max r={max(vals)}")
        plt.show()
        return


class ActionBuffer:
    def __init__(self, n, capacity, state_dim, M, seed):
        self.id = n
        self.state_dim = state_dim
        self.capacity = capacity
        self.M = M
        self.ef_construction = 200
        self._tree = hnswlib.Index(space="l2", dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity,
                              M=self.M,
                              ef_construction=self.ef_construction,
                              random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.times_list = []
        self.raw_states = []
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.seed = seed

    def __getstate__(self):
        # pickle everything but the hnswlib indexes
        self._tree.save_index(f"saves/index_{self.id}.bin")
        return dict((k, v) for (k, v) in self.__dict__.items() if k != "_tree")

    def __setstate__(self, d):
        self.__dict__ = d
        self._tree = hnswlib.Index(space="l2", dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    def normalize(self, state):
        """can be single or list of states - will be broadcast"""
        # print(f"before: {state}, after:{np.subtract(state, self.mean)/self.std}")
        return np.subtract(state, self.mean) / self.std

    def update_normalization(self, mean, std):
        self.mean = mean
        self.std = std
        self._tree = hnswlib.Index(space="l2", dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=self.capacity,
                              ef_construction=self.ef_construction,
                              M=self.M,
                              random_seed=self.seed)
        self._tree.add_items(self.normalize(self.raw_states))

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(self.normalize(state), k=k)

    def add(self, state, value, time):
        normalized_state = self.normalize(state)
        self._tree.add_items(normalized_state)
        self.raw_states.append(state)
        self.values_list.append(value)
        self.times_list.append(time)

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values_list)
