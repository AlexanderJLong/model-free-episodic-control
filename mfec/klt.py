#!/usr/bin/env python3

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import umap


class KLT:
    def __init__(self, actions, buffer_size, k, state_dim, obv_dim, distance, seed):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.buffers = tuple(
            [ActionBuffer(a, self.buffer_size, state_dim, distance, seed) for a in actions])
        self.k = k
        self.obv_dim = obv_dim  # dimentionality of origional data

    def gaus(self, x, sig):
        return np.exp(-np.square(x/sig)/2)

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

        k = np.sqrt(current_size)//1
        k = int(max(k, 1))
        neighbors, dists = buffer.find_neighbors(state, k)
        # Strip batch dim. dists is already ordered. Note it is square of l2 norm.

        dists = np.sqrt(dists[0]) #TODO: normalize?
        neighbors = neighbors[0]

        values = [buffer.values_list[n] for n in neighbors]

        # If all dists are 0, need to forget earliest estimate in that buffer to continue learning
        if np.all(dists == 0):
            # All visited, just return mean since dists is 0
            weighted_reward = np.mean(values)
            weighted_dist = 0
            buffer.remove(neighbors[0])  # smallest id is earliest sample and neighbours is ordered
            # TODO isn't it ordered by dist?
        else:
            # Not all visited so do a weighted mean
            weighted_dist = np.mean(dists)
            w = self.gaus(dists, sig=weighted_dist * 1.5)
            weighted_reward = np.mean(values*w) / sum(w)

        return weighted_reward, weighted_dist

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

    def plot3d(self,):
        fig = plt.figure()
        reducer = umap.UMAP(n_neighbors=200, n_components=2)

        fig.set_tight_layout(True)
        rows = 4
        for i, buffer in enumerate(self.buffers):
            ax = fig.add_subplot(rows, len(self.buffers)//rows+1, i+1)

            states = np.asarray(buffer.get_states())
            embeddings = reducer.fit_transform(states)
            vals = np.asarray(buffer.values_list)
            ax.scatter(embeddings[:, 1], embeddings[:, 0], c=vals)
            ax.set(xlabel="Vel")
            ax.set(ylabel="Angle")

            ax.set(title=f"max r={max(vals)}")
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
