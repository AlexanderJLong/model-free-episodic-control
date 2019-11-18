#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class QEC:
    def __init__(self, actions, buffer_size, k, kernel_type, kernel_width, state_dim):
        self.buffers = tuple([ActionBuffer(buffer_size, state_dim) for _ in actions])
        self.k = k
        self.mu = np.zeros(state_dim)  # offset
        self.sig = np.ones(state_dim)  # scale
        self.kernel_width = kernel_width
        self.kernel_type = kernel_type

    def estimate(self, state, action):
        """Return the estimated value of the given state"""

        buffer = self.buffers[action]
        if len(buffer) < self.k:
            return float("inf")

        neighbors, dists = buffer.find_neighbors(state, self.k)
        # Strip batch dim. Note dists is already ordered.
        dists = dists[0]
        neighbors = neighbors[0]

        # Identical state found
        if dists[0] == 0:
            return buffer.values[neighbors[0]]

        w = [1 / d for d in dists]

        value = 0
        for i, neighbor in enumerate(neighbors):
            value += w[i] * buffer.values[neighbor]

        return value / sum(w)

    def update(self, state, action, value):
        buffer = self.buffers[action]
        buffer.add(state, value)

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
            if len(self.buffers[0].values) < 10:
                return
            num_actions = len(self.buffers)
            cols = 4
            rows = num_actions // cols + 1
            max_r = max([max(b.values) for b in self.buffers])
            for i in range(num_actions):
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')

                data = self.buffers[i]
                states = np.asarray(data.get_states())[::]
                if len(states) < 1:
                    return
                vals = np.asarray(data.values)[::]
                ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=vals, vmax=max_r)

            plt.show()
        return


class ActionBuffer:
    def __init__(self, capacity, state_dim):
        self.state_dim = state_dim
        self.capacity = capacity
        self._tree = hnswlib.Index(space='l2', dim=state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity, ef_construction=200, M=16)
        self.values = []

    def reset(self, data):
        """Reset the buffer with just the data provided"""
        self._tree = hnswlib.Index(space='l2', dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=self.capacity, ef_construction=40, M=16)
        self._tree.add_items(data)

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(np.asarray(state), k=k)

    def add(self, state, value):
        if self.values: # at least one sample in buffer
            idx, dist = self.find_neighbors(state, 1)
            idx = idx[0][0]
            dist = dist[0][0]
        else:
            dist = 10

        if dist < 1e-6:
            self.values[idx] = max(value, self.values[idx])
        else:
            self.values.append(value)
            self._tree.add_items(state)

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values)
