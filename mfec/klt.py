#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, lr, time_sig, seed):
        self.buffer_size = buffer_size
        self.buffers = tuple(
            [ActionBuffer(a, self.buffer_size, state_dim, distance, lr, time_sig, seed) for a in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data
        self.state_dim = state_dim
        self.time_horizon = time_sig

    def gaus(self, x, h):
        return 1 / (h * 2 * np.pi) * np.exp(-np.square(x / h))

    def gaus_2d(self, x, y, sig1, sig2):
        return np.exp(-(np.square(x / sig1) + np.square(y / sig2)))

    def reconstruct_trees(self, u, sig):
        for b in self.buffers:
            b.reconstruct(u=u, sig=sig)

    def estimate(self, state, action, count_weight):
        """Return the estimated value of the given state"""

        buffer = self.buffers[action]

        n = len(buffer)
        if n == 0:
            return 1e6, 0  # TODO: not neat
        k = min(self.k, n)  # the len call might slow it down a bit
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. Note dists is already ordered.
        dists = dists[0]
        neighbors = neighbors[0]

        norms = np.sqrt(dists)

        # print(dists, neighbors, buffer.values_array, action)
        # never seen before so estimate
        values = np.asarray([buffer.values_list[n] for n in neighbors])
        times = np.asarray([buffer.times_list[n] for n in neighbors])

        density = norms[-1] / k  # average dist to find one sample
        w = self.gaus_2d(norms, times, sig1=density * k + 0.01, sig2=self.time_horizon)

        w_sum = np.sum(w)
        weighted_reward = np.dot(w, values) / w_sum

        return weighted_reward, density

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

    def plot3d(self, both, diff):
        fig = plt.figure()
        fig.set_tight_layout(True)
        if diff and both:
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            axes = [ax1, ax2]
            for i, ax in enumerate(axes):
                data = self.buffers[i]
                states = np.asarray(data.get_states())
                vals = np.asarray(data.values)
                ax.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals)
                ax.set(xlabel="Vel")
                ax.set(ylabel="Angle")
                ax.set(zlabel="Position")

                ax.set(title=f"max r={max(vals)}")

            states = np.random.rand(5000, 4) * 5 - 2
            states[:, -1] = 0
            vals = []
            for s in states:
                vals.append(self.estimate(s, 1, 0) - self.estimate(s, 0, 0))

            # force normalization between certain range and make sure its symetric
            vals[0] = max(max(vals), -min(vals))
            vals[1] = min(-max(vals), min(vals))
            ax3.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals, cmap="bwr")
            ax3.set(xlabel="Vel")
            ax3.set(ylabel="Angle")
            ax3.set(zlabel="Position")
            ax3.set(title=f"max={max(vals):.2f}, min={min(vals):.2f}")
            plt.show()
            return

        elif diff:
            ax = fig.add_subplot(111, projection='3d')
            states = np.random.rand(5000, 4) * 5 - 2
            vals = []
            for s in states:
                vals.append(self.estimate(s, 1, 0))
            ax.scatter(states[:, 1], states[:, 2], states[:, 0], c=vals)

            ax.set(xlabel="Vel")
            ax.set(ylabel="Angle")
            ax.set(zlabel="Position")
            plt.show()
            return

        else:
            if len(self.buffers[0].values_list) < 10:
                return
            num_actions = len(self.buffers)
            cols = 4
            rows = num_actions // cols + 1
            max_r = max([max(b.values_list) for b in self.buffers])
            for i in range(num_actions):
                ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

                data = self.buffers[i]
                states = np.asarray(data.get_states())[::]
                if len(states) < 1:
                    return
                vals = np.asarray(data.values)[::]
                ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=vals, vmax=max_r)

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
        self.M = 25
        self.ef_construction = 200
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity,
                              M=self.M,
                              ef_construction=self.ef_construction,
                              random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.
        self.times_list = []
        self.raw_states = []
        self.seed = seed

    def __getstate__(self):
        # pickle everything but the hnswlib indexes
        self._tree.save_index(f"saves/index_{self.id}.bin")
        return dict((k, v) for (k, v) in self.__dict__.items() if k != "_tree")

    def __setstate__(self, d):
        self.__dict__ = d
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    def reconstruct(self, u, sig):
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=self.capacity,
                              ef_construction=self.ef_construction,
                              M=self.M,
                              random_seed=self.seed)

        #TODO: Inter or intra buffer stats?
        states = (np.asarray(self.raw_states) - u) / sig  # normalize
        self._tree.add_items(states)

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(state, k=k)

    def add(self, state, value, time):
        self.values_list.append(value)
        self.raw_states.append(state[0])
        self.times_list.append(time)

        return

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values_list)
