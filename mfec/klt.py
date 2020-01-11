#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, seed):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.buffers = tuple(
            [ActionBuffer(a, self.buffer_size, state_dim, distance, seed) for a in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data

    def estimate(self, state, action):
        """
        Return:
         1) the estimated value of the given state
         2) a measure of distance to surrounding samples

         According to this measure we don't care once we have more than k samples on a point, it just gets treated
         the same as we add more. So need a reasonably high k.
         """

        buffer = self.buffers[action]
        current_size = len(buffer)
        if current_size == 0:
            return 0, 0  # Maybe better to just signal the buffer is empty than assigning large dist

        k = min(self.k, current_size)
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. dists is already ordered. Note it is square of l2 norm.

        dists = np.sqrt(dists[0])
        neighbors = neighbors[0]

        counts = np.sum(dists == 0)

        values = [buffer.values_list[n] for n in neighbors]
        weighted_reward = np.mean(values)

        # If all dists are 0, need to forget earliest estimate in that buffer to continue learning
        if counts == self.k:
            buffer.remove(neighbors[0])  # smallest id is earliest sample and neighbours is ordered
        if counts == 0:
            # never visited so return mean of dists inverse.
            counts = 1/np.min(dists)

        return weighted_reward, counts

    def update(self, state, action, value):
        # print("updating", action)
        buffer = self.buffers[action]
        buffer.add(state, value)

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
    def __init__(self, n, capacity, state_dim, distance, seed):
        self.id = n
        self.state_dim = state_dim
        self.capacity = capacity
        self.distance = distance
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)  # possible options are l2, cosine or ip
        self._tree.init_index(max_elements=capacity, M=100, random_seed=seed)
        self.values_list = []  # true values - this is the object that is updated.

    def __getstate__(self):
        # pickle everything but the hnswlib indexes
        self._tree.save_index(f"saves/index_{self.id}.bin")
        return dict((k, v) for (k, v) in self.__dict__.items() if k != "_tree")

    def __setstate__(self, d):
        self.__dict__ = d
        self._tree = hnswlib.Index(space=self.distance, dim=self.state_dim)
        self._tree.load_index(f"saves/index_{self.id}.bin")

    def find_neighbors(self, state, k):
        """Return idx, dists"""
        return self._tree.knn_query(state, k=k)

    def add(self, state, value):
        self.values_list.append(value)
        self._tree.add_items(state)
        return

    def remove(self, idx):
        """Remove a sample"""
        self._tree.mark_deleted(idx)

    def get_states(self):
        return self._tree.get_items(range(0, len(self)))

    def __len__(self):
        return len(self.values_list)
