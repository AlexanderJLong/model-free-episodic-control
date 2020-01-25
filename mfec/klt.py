#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, lr, time_sig, seed):
        self.buffer_size = buffer_size
        self.buffers = tuple(
            [ActionBuffer(n=a,
                          capacity=self.buffer_size,
                          state_dim=state_dim,
                          distance=distance,
                          lr=lr,
                          agg_dist=0.01,
                          seed=seed,
                          ) for a in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data
        self.time_horizon = time_sig

    def gaus(self, x, sig):
        return np.exp(-np.square(x / sig) / 2)

    def gaus_2d(self, x, y, sig1, sig2):
        return np.exp(-(np.square(x / sig1) + np.square(y / sig2)))

    def update_normalization(self, mean, std):
        for b in self.buffers:
            b.update_normalization(mean=mean, std=std)

    def estimate(self, state, action, count_weight):
        """Return the estimated value of the given state"""
        buffer = self.buffers[action]

        if len(buffer) == 0:
            return 1e6, 0, 0
        k = min(self.k, len(buffer))  # the len call might slow it down a bit
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. Note dists is already ordered.
        dists = dists[0]
        neighbors = neighbors[0]

        # print(dists, neighbors, buffer.values_array, action)
        # never seen before so estimate
        values = np.asarray([buffer.values_list[n] for n in neighbors])
        counts = np.asarray([buffer.counts_list[n] for n in neighbors])
        times = np.asarray([buffer.times_list[n] for n in neighbors])
        # counts = buffer.counts_array[neighbors]

        # Convert to l2norm, normalize by original dimensionality so dists have a consistent
        # range, but make sure they're always still > 1 because of w=1/d
        norms = np.sqrt(dists)
        # norms[norms == 0] = 1
        # w = np.divide(1., norms)  # Get inverse distances as weights
        h = np.mean(norms) / 2 if np.min(
            norms) != 0 else 1  # This reduces to only considering exact matches when they are there.
        w = self.gaus_2d(norms, times, sig1=h, sig2=self.time_horizon)

        w_sum = np.sum(w)
        weighted_reward = np.dot(w, values) / w_sum
        weighted_count = np.dot(w, counts) / w_sum

        return weighted_reward, weighted_count, 0

    def update(self, state, action, value, time):
        # print("updating", action)
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
    def __init__(self, n, capacity, state_dim, distance, lr, agg_dist, seed):
        self.id = n
        self.agg_dist = agg_dist
        self.state_dim = state_dim
        self.lr = lr
        self.capacity = capacity
        self.distance = distance
        self.M = 200
        self.ef_construction = 200
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity,
                              M=self.M,
                              ef_construction=self.ef_construction,
                              random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.counts_list = []
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
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    def normalize(self, state):
        "can be single or list of states - will be broadcast"
        # print(f"before: {state}, after:{np.subtract(state, self.mean)/self.std}")
        return np.subtract(state, self.mean) / self.std

    def update_normalization(self, mean, std):
        self.mean = mean
        self.std = std
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
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
        if not self.values_list:  # buffer empty, just add
            self._tree.add_items(normalized_state)
            self.raw_states.append(state)
            self.values_list.append(value)
            self.counts_list.append(1)
            self.times_list.append(time)
            return

        idx, dist = self.find_neighbors(normalized_state, 1)
        idx = idx[0][0]
        dist = dist[0][0]

        if dist < self.agg_dist:
            # Existing state, update and return
            self.counts_list[idx] += 1
            self.values_list[idx] = 0.9 * self.values_list[idx] + 0.1 * value
            self.times_list[idx] = time
        else:
            self.values_list.append(value)
            self._tree.add_items(normalized_state)
            self.raw_states.append(state)
            self.counts_list.append(1)
            self.times_list.append(time)

        return

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values_list)
